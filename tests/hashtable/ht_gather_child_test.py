import os
import unittest
from collections import defaultdict
from typing import Tuple

import torch
import torch.distributed as dist
import torch.testing._internal.common_utils as common

from recis.nn.functional import fused_ops
from recis.nn.initializers import ConstantInitializer
from recis.nn.modules.embedding import EmbeddingOption
from recis.nn.modules.embedding_engine import EmbeddingEngine


def nested_dict():
    return defaultdict(nested_dict)


def ts_equal(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    t1 = t1.cuda()
    t2 = t2.cuda()
    if t1.shape != t2.shape:
        print(
            f"Tensors have different shapes: t1.shape={t1.shape}, t2.shape={t2.shape}"
        )
        return False
    sorted_t1, _ = torch.sort(t1)
    sorted_t2, _ = torch.sort(t2)
    is_equal = torch.equal(sorted_t1, sorted_t2)
    if not is_equal:
        print(
            f"Tensors do not contain the same set of elements, {sorted_t1}, {sorted_t2}"
        )
        for i in range(sorted_t1.shape[0]):
            if sorted_t1[i] != sorted_t2[i]:
                print(
                    f"  - At index {i}: t1 has value {sorted_t1[i].item()}, but t2 has value {sorted_t2[i].item()}"
                )
    return is_equal


def compare_snapshot(
    mapping1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mapping2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    ids1, indices1, global_embs1 = mapping1
    ids2, indices2, global_embs2 = mapping2

    if ids1.size(0) != ids2.size(0):
        return False
    if ids1.size(0) == 0:
        return True

    # compare ids -> index
    sorted_ids1, sort_idx1 = torch.sort(ids1)
    sorted_ids2, sort_idx2 = torch.sort(ids2)
    if not torch.equal(sorted_ids1, sorted_ids2):
        return False
    sorted_indices1 = indices1[sort_idx1]
    sorted_indices2 = indices2[sort_idx2]
    if not torch.equal(sorted_indices1, sorted_indices2):
        return False
    # compare index -> embs
    return torch.allclose(
        global_embs1[sorted_indices1],
        global_embs2[sorted_indices2],
        rtol=rtol,
        atol=atol,
    )


class HashTableGatherChildTest(unittest.TestCase):
    DEVICE = None

    @classmethod
    def setUpClass(cls):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        cls.DEVICE = os.getenv("TEST_DEVICE", "cuda")
        dist.init_process_group()

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def setIds(self):
        self.ids = {
            "fea_1": torch.tensor(
                [12, 13, 14, 15, 16], dtype=torch.int64, device="cuda"
            ),
            "fea_2": torch.tensor(
                [27, 28, 29, 30, 31, 32, 33], dtype=torch.int64, device="cuda"
            ),
            "fea_3": torch.tensor([12, 99], dtype=torch.int64, device="cuda"),
            "fea_4": torch.tensor([999999, 88], dtype=torch.int64, device="cuda"),
        }
        self.embs = {
            "emb_1": torch.full((6, 8), 1.0, dtype=torch.float32, device="cuda"),
            "emb_2": torch.full((7, 8), 2.0, dtype=torch.float32, device="cuda"),
            "emb_3": torch.full((2, 8), 4.0, dtype=torch.float32, device="cuda"),
        }

    def setFeas(self):
        self.group_a = nested_dict()
        self.group_a["ht1"]["emb_opt"] = {
            "fea_1": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht1",
                combiner="sum",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
            ),
            "fea_3": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht1",
                combiner="mean",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
            ),
        }
        self.group_a["ht2"]["emb_opt"] = {
            "fea_2": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht2",
                combiner="mean",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
            )
        }
        self.group_a["ht3"]["emb_opt"] = {
            "fea_4": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht3",
                combiner="mean",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
            )
        }

    def setEmbeddingEngine(self):
        from collections import ChainMap

        ee_init = dict(
            ChainMap(
                self.group_a["ht3"]["emb_opt"],
                self.group_a["ht2"]["emb_opt"],
                self.group_a["ht1"]["emb_opt"],
            )
        )
        self.ee = EmbeddingEngine(ee_init)

    def setHtInfo(self):
        self.tables = nested_dict()
        group_to_fea = {
            "group_a": "fea_1",
        }
        for group_name, fea_name in group_to_fea.items():
            ht_key = self.ee._fea_to_ht[fea_name]
            self.tables[group_name]["ht"] = self.ee._ht[ht_key]._hashtable

    def setEncodeIds(self):
        self.encode_ids = defaultdict(str)
        self.feas = ["fea_1", "fea_2", "fea_3", "fea_4"]
        for fea in self.feas:
            self.encode_ids[fea] = self.ee._fea_to_group[fea].encode_id(fea)

    def setUp(self):
        self.setIds()
        self.setFeas()
        self.setEmbeddingEngine()
        self.setHtInfo()
        self.setEncodeIds()

    def testGatherChild(self):
        coalesced_ids = torch.unique(
            fused_ops.fused_ids_encode_gpu(
                [
                    self.ids["fea_1"],
                    self.ids["fea_2"],
                    self.ids["fea_3"],
                    self.ids["fea_4"],
                ],
                [
                    self.encode_ids["fea_1"],
                    self.encode_ids["fea_2"],
                    self.encode_ids["fea_3"],
                    self.encode_ids["fea_4"],
                ],
            )
        )
        coalesced_embs = torch.cat(list(self.embs.values()))
        self.tables["group_a"]["ht"].insert(coalesced_ids, coalesced_embs)
        # check child emb
        for emb_id in range(1, 4):
            self.assertTrue(
                torch.equal(
                    self.embs[f"emb_{emb_id}"],
                    self.tables["group_a"]["ht"].embeddings(f"ht{emb_id}"),
                )
            )
        # check child id
        self.assertTrue(
            ts_equal(
                torch.unique(torch.cat([self.ids["fea_1"], self.ids["fea_3"]])),
                self.tables["group_a"]["ht"].ids("ht1"),
            )
        )
        self.assertTrue(
            ts_equal(
                fused_ops.fused_ids_encode_gpu(
                    [self.ids["fea_2"]], [self.encode_ids["fea_2"]]
                ),
                self.tables["group_a"]["ht"].ids("ht2"),
            )
        )
        self.assertTrue(
            ts_equal(
                fused_ops.fused_ids_encode_gpu(
                    [self.ids["fea_4"]], [self.encode_ids["fea_4"]]
                ),
                self.tables["group_a"]["ht"].ids("ht3"),
            )
        )
        # check whole map
        self.assertTrue(
            self.tables["group_a"]["ht"].embeddings().size(0) == coalesced_ids.numel()
        )
        self.assertTrue(self.tables["group_a"]["ht"].raw_embeddings().size(0) == 10240)
        self.assertTrue(ts_equal(self.tables["group_a"]["ht"].ids(), coalesced_ids))
        # check snapshot
        snapshot_full = self.tables["group_a"]["ht"].snap_shot()
        ids_full, index_full = self.tables["group_a"]["ht"].ids_map()
        _, embedding = self.tables["group_a"]["ht"]._hashtable_impl.embedding_lookup(
            ids_full, True
        )
        compare_snapshot(snapshot_full, (ids_full, index_full, embedding))
        # check ids_embs_map
        for emb_id in range(1, 4):
            ids, emba = self.tables["group_a"]["ht"].ids_embeddings(f"ht{emb_id}")
            _, embb = self.tables["group_a"]["ht"]._hashtable_impl.embedding_lookup(
                ids, True
            )
            self.assertTrue(torch.equal(emba, embb))


if __name__ == "__main__":
    unittest.main()

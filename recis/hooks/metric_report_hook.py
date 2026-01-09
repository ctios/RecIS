import time
from dataclasses import dataclass
from typing import Optional

from recis.hooks.hook import Hook
from recis.metrics.metric_reporter import (
    EVAL_QPS_NAME,
    FLOPS_NAME,
    HT_ALL_SLOT_BYTES,
    HT_ALLOCATOR_ID_ACT_SIZE,
    HT_ALLOCATOR_ID_TOTAL_SIZE,
    HT_EMB_BYTES,
    HT_ID_ACT_SIZE,
    HT_ID_TOTAL_BYTES,
    HT_ID_TOTAL_SIZE,
    PREPARE_NAME,
    QPS_NAME,
    TRAIN_QPS_NAME,
    MetricReporter,
)
from recis.nn.modules.hashtable import filter_out_sparse_param


@dataclass
class ReportArguments:
    interval_step: int = 100


class MetricReportHook(Hook):
    _internal_profs = {
        FLOPS_NAME: 0,
    }

    def __init__(self, model, report_args: Optional[ReportArguments] = None):
        super().__init__()
        if report_args is None:
            report_args = ReportArguments()
        self.model = model
        self.hashtables = filter_out_sparse_param(model)
        self.args = report_args
        self.steps = 0
        self.train_steps = 0
        self.eval_steps = 0
        self.interval_time = time.time()
        self.step_time = time.time()
        self.activate = False  # indicate whether current step is activate to report

    def _reset(self):
        self.train_steps = 0
        self.eval_steps = 0
        self.interval_time = time.time()

    def _report_metrics(self):
        # qps, train qps, eval qps
        spend_time = time.time() - self.interval_time
        qps = self.args.interval_step / spend_time
        train_qps = self.train_steps / spend_time
        eval_qps = self.eval_steps / spend_time
        flops_total = (
            self.__class__._internal_profs.get(FLOPS_NAME, 0)
            * self.args.interval_step
            / spend_time
        )
        MetricReporter.report(QPS_NAME, qps, {"recis_qps_type": QPS_NAME})
        MetricReporter.report(QPS_NAME, train_qps, {"recis_qps_type": TRAIN_QPS_NAME})
        MetricReporter.report(QPS_NAME, eval_qps, {"recis_qps_type": EVAL_QPS_NAME})
        MetricReporter.report(FLOPS_NAME, flops_total, {"recis_flops_type": FLOPS_NAME})

        # hashtable
        for ht_name, ht in self.hashtables.items():
            act_num, total_num = ht.id_info()
            MetricReporter.report(HT_ID_ACT_SIZE, act_num, {"recis_ht_name": ht_name})
            MetricReporter.report(
                HT_ID_TOTAL_SIZE, total_num, {"recis_ht_name": ht_name}
            )
            allocator_act_num, allocator_total_num = ht.allocator_id_info()
            MetricReporter.report(
                HT_ALLOCATOR_ID_ACT_SIZE, allocator_act_num, {"recis_ht_name": ht_name}
            )
            MetricReporter.report(
                HT_ALLOCATOR_ID_TOTAL_SIZE,
                allocator_total_num,
                {"recis_ht_name": ht_name},
            )
            total_mem = ht.id_memory_info()
            MetricReporter.report(
                HT_ID_TOTAL_BYTES, total_mem, {"recis_ht_name": ht_name}
            )
            emb_mem, total_mem = ht.emb_memory_info()
            MetricReporter.report(HT_EMB_BYTES, emb_mem, {"recis_ht_name": ht_name})
            MetricReporter.report(
                HT_ALL_SLOT_BYTES, total_mem, {"recis_ht_name": ht_name}
            )

    def before_step(self, is_train=True, *args, **kwargs):
        if self.args.interval_step is None:
            return
        if self.steps % self.args.interval_step != 0:
            return
        self.step_time = time.time()
        self.activate = True
        MetricReporter.set_reportable(True)

    def after_step(self, is_train=True, *args, **kwargs):
        self.steps += 1
        if is_train:
            self.train_steps += 1
        else:
            self.eval_steps += 1
        if not self.activate:
            return
        self._report_metrics()
        self._reset()
        MetricReporter.set_reportable(False)
        self.activate = False

    def out_off_data(self, *args, **kwargs):
        self._reset()
        MetricReporter.set_reportable(False)
        self.activate = False

    def after_data(self, is_train=True, *args, **kwargs):
        if self.activate:
            eclapsed_time = (time.time() - self.step_time) * 1000
            MetricReporter.report(PREPARE_NAME, eclapsed_time)

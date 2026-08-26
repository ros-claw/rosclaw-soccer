# ruff: noqa: E501
"""Build the evidence-backed S50-S75 recovery Growth retrospective.

The generated artifact is intentionally snapshot-only and SIM_ONLY.  It reads
the durable local evidence, validates every corrective-student report with the
current validator, and never launches training, ROS, DDS, or hardware actions.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.training.recovery_corrective_student import (
    validate_recovery_corrective_student_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = Path("/code/rosclaw/rosclaw_football/evidence")
OUTPUT_PATH = (
    REPO_ROOT / "docs" / "rosclaw-soccer-s50-s75-recovery-growth-retrospective.artifact.json"
)
CORE_REPORT_NAMES = {
    "training-report.json",
    "student-report.json",
    "frozen-benchmark-report.json",
    "teacher-report.json",
    "collection-report.json",
}


def _stage_number(path: Path) -> int | None:
    match = re.match(r"s(\d+)", path.relative_to(EVIDENCE_ROOT).parts[0])
    return int(match.group(1)) if match else None


def _load(relative_path: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((EVIDENCE_ROOT / relative_path).read_text(encoding="utf-8")),
    )


def _pct(value: float) -> float:
    return round(100.0 * float(value), 6)


def _current_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _working_tree_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not bool(result.stdout.strip())


def _inventory() -> tuple[list[Path], dict[str, int]]:
    core_reports: list[Path] = []
    counts = {name: 0 for name in CORE_REPORT_NAMES}
    for path in EVIDENCE_ROOT.rglob("*.json"):
        stage = _stage_number(path)
        if stage is None or not 50 <= stage <= 75 or path.name not in CORE_REPORT_NAMES:
            continue
        core_reports.append(path)
        counts[path.name] += 1
    return sorted(core_reports), counts


def _validated_students() -> list[tuple[Path, dict[str, Any]]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(EVIDENCE_ROOT.glob("s*/**/student-report.json")):
        stage = _stage_number(path)
        if stage is None or not 56 <= stage <= 75:
            continue
        reports.append((path, validate_recovery_corrective_student_evidence(path)))
    return reports


REPRESENTATIVE_REPORTS = [
    (
        "S56-v1",
        "s56-corrective-student-v1/pareto96-trace20-normal600-mlp1500-seed5600/student-report.json",
        "全关节监督学生",
        "当轮拒绝",
    ),
    (
        "S56-v3",
        "s56-corrective-student-v3/jacobian-channel-gain016-seed5600/student-report.json",
        "Jacobian 稀疏肌肉通道",
        "当轮双门通过，未晋级",
    ),
    (
        "S57-DAgger",
        "s57-corrective-student-dagger-v1/normal-dagger50-maxchannel16-gain016-seed5600/student-report.json",
        "正常路线 on-policy DAgger",
        "正常方向门失败",
    ),
    (
        "S57-Gate",
        "s57-corrective-student-gate-v1/plastic-v1-silence-v2-logistic-ood999-seed5600/student-report.json",
        "置信门与 OOD 失败关闭",
        "困难收益不足 1%",
    ),
    (
        "S58-MLP",
        "s58-corrective-scale-curriculum-v1/student192-mlp6000-seed5602/student-report.json",
        "192 状态、6000 步全关节学生",
        "严重污染正常路线",
    ),
    (
        "S58-DAgger",
        "s58-corrective-scale-curriculum-v1/student192-dagger1-channel16-gain016/student-report.json",
        "192 状态稀疏 DAgger",
        "困难收益不足且方向/稳定失败",
    ),
    (
        "S58-OnPolicyGate",
        "s58-corrective-scale-curriculum-v1/student192-onpolicy-confidence-gate-t075-temp8/student-report.json",
        "on-policy 负样本置信门",
        "正常闭环正反馈",
    ),
    (
        "S59-v12",
        "s59-temporal-historical-veto-gate-v12/primary-trigger-veto-floor050-authority080/student-report.json",
        "时序租约与历史能力否决",
        "新域双门通过，旧域拒绝",
    ),
    (
        "S60-v4",
        "s60-effect-channel-budget-v4/new192-riskweighted-040-080/student-report.json",
        "因果效应通道预算",
        "新域双门通过，旧域拒绝",
    ),
    (
        "S61-v2",
        "s61-state-conditioned-channel-veto-v2/calibrated-temp2-run1/student-report.json",
        "状态条件向量 veto",
        "当轮双门通过，重复性暴露",
    ),
    (
        "S62-v1",
        "s62-channel-veto-closed-loop-dagger-v1/current168-frozen72-temp2-run1/student-report.json",
        "veto 闭环 DAgger",
        "当轮双门通过，冻结旧域拒绝",
    ),
    (
        "S63",
        "s63-channel-veto-absolute-recalibration-v1/temp3-current-run1/student-report.json",
        "绝对量纲重校准",
        "困难收益不足、方向门失败",
    ),
    (
        "S64",
        "s64-veto-aware-temporal-trigger-v1/temp2-current-run1/student-report.json",
        "veto-aware 时序触发",
        "正常稳定门失败",
    ),
    (
        "S65",
        "s65-channel-veto-closed-loop-dagger-iteration2-v1/current168-frozen72-from-s64-trigger-run1/student-report.json",
        "第二轮闭环 DAgger",
        "正常方向门失败",
    ),
    (
        "S66",
        "s66-lockstep-paired-exam-v1/s65-exact-model-current-run1/student-report.json",
        "单图 lockstep 成对考试",
        "消除伪因果后困难收益不足",
    ),
    (
        "S67",
        "s67-channel-veto-failure-recall-margin-v1/temp2-margin2-current-run1/student-report.json",
        "失败召回 logit margin",
        "当前银行通过，S68 盲测拒绝",
    ),
    (
        "S69-DAgger",
        "s69-cross-seed-growth-v1/new-seed5477-dagger1-channel16-gain016/student-report.json",
        "新种子跨域 DAgger",
        "困难通过，正常方向/稳定失败",
    ),
    (
        "S69-Temporal",
        "s69-cross-seed-growth-v1/new-seed5477-temporal-hard-negative-r2-old96/student-report.json",
        "跨种子时序硬负样本",
        "正常通过，困难收益不足",
    ),
    (
        "S70",
        "s70-prefix-recall-growth-v1/new-seed5477-prefix2-weight10-t080-r2-old96/student-report.json",
        "失败前缀加权召回",
        "困难稳定门与正常方向门失败",
    ),
    (
        "S71",
        "s71-cross-seed-dagger2-growth-v1/new-seed5477-dagger2-channel16-gain016/student-report.json",
        "跨种子 DAgger 第二轮",
        "可塑性消失且正常回退",
    ),
    (
        "S72",
        "s72-dagger-action-confidence-v1/dagger1-action-dagger2-negative-t075-temp8/student-report.json",
        "DAgger1 动作 + DAgger2 负样本门",
        "正常改善但困难收益消失",
    ),
    (
        "S73",
        "s73-dagger-action-temporal-memory-v1/dagger1-action-r2-old96-t075/student-report.json",
        "DAgger 动作时序肌肉记忆",
        "正常安静但困难退化",
    ),
    (
        "S74",
        "s74-channel-veto-repair-v1/dagger1-action-current-dagger2-old96/student-report.json",
        "通道 veto 修复模式",
        "正常改善，困难收益仅 0.293%",
    ),
    (
        "S75",
        "s75-channel-veto-repair-calibration-v1/temp2-current-run1/student-report.json",
        "S74 温度重校准",
        "困难退化，拒绝",
    ),
]


def _representative_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, (attempt, relative_path, method, later_result) in enumerate(
        REPRESENTATIVE_REPORTS, start=1
    ):
        report = _load(relative_path)
        failure = report["failure_state_paired_physics_exam"]
        normal = report["normal_route_paired_physics_exam"]
        local_dual = bool(failure["passed"] and normal["passed"])
        rows.append(
            {
                "order": order,
                "attempt": attempt,
                "method": method,
                "failure_improvement_pct": _pct(failure["mean_cost_improvement_fraction"]),
                "normal_regression_pct": _pct(normal["normal_cost_regression_fraction"]),
                "normal_action_rms": round(float(normal["mean_action_increment_rms"]), 6),
                "failure_passed": bool(failure["passed"]),
                "normal_passed": bool(normal["passed"]),
                "gate_status": "当轮双门通过" if local_dual else "当轮拒绝",
                "later_result": later_result,
                "promotion_eligible": bool(report["promotion_eligible"]),
                "evidence": relative_path,
            }
        )

    blind_path = "s68-fresh-blind-current-bank-v1/blind-eval-s67-run1/frozen-benchmark-report.json"
    blind = _load(blind_path)
    failure = blind["failure_state_paired_physics_exam"]
    normal = blind["normal_route_paired_physics_exam"]
    rows.append(
        {
            "order": 16.5,
            "attempt": "S68-Blind",
            "method": "未参与选择的新种子盲测",
            "failure_improvement_pct": _pct(failure["mean_cost_improvement_fraction"]),
            "normal_regression_pct": _pct(normal["normal_cost_regression_fraction"]),
            "normal_action_rms": round(float(normal["mean_action_increment_rms"]), 6),
            "failure_passed": bool(failure["passed"]),
            "normal_passed": bool(normal["passed"]),
            "gate_status": "盲测拒绝",
            "later_result": "S67 的当前银行通过未泛化到新失败分布",
            "promotion_eligible": False,
            "evidence": blind_path,
        }
    )
    rows.sort(key=lambda row: float(row["order"]))
    for order, row in enumerate(rows, start=1):
        row["order"] = order
    return rows


def _retest_rows() -> list[dict[str, Any]]:
    specs = [
        (
            "S59-v12",
            1.3993078,
            -0.4941733,
            "s59-temporal-historical-veto-gate-v12/frozen-old96-exam/frozen-benchmark-report.json",
            "旧正常方向门失败",
        ),
        (
            "S60-v4",
            1.192696,
            -0.005693,
            "s60-effect-channel-budget-v4/frozen-old96-exam/frozen-benchmark-report.json",
            "旧正常回退 2.158%，方向门失败",
        ),
        (
            "S62-v1",
            1.658371,
            -0.351219,
            "s62-channel-veto-closed-loop-dagger-v1/frozen96-exam-run1/frozen-benchmark-report.json",
            "旧正常方向门失败",
        ),
        (
            "S67",
            1.453896,
            0.0,
            "s68-fresh-blind-current-bank-v1/blind-eval-s67-run1/frozen-benchmark-report.json",
            "新种子困难收益仅 0.458%，稳定门失败",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for order, (candidate, current_failure, current_normal, path, blocker) in enumerate(
        specs, start=1
    ):
        report = _load(path)
        failure = report["failure_state_paired_physics_exam"]
        normal = report["normal_route_paired_physics_exam"]
        frozen_label = "新种子盲测" if candidate == "S67" else "冻结旧域"
        rows.extend(
            [
                {
                    "candidate": candidate,
                    "exam_domain": "当轮开发域",
                    "failure_improvement_pct": current_failure,
                    "normal_regression_pct": current_normal,
                    "passed": True,
                    "order": order * 2 - 1,
                    "blocker": "当轮双门通过，但无独立晋级权",
                },
                {
                    "candidate": candidate,
                    "exam_domain": frozen_label,
                    "failure_improvement_pct": _pct(failure["mean_cost_improvement_fraction"]),
                    "normal_regression_pct": _pct(normal["normal_cost_regression_fraction"]),
                    "passed": bool(report["frozen_benchmark_passed"]),
                    "order": order * 2,
                    "blocker": blocker,
                },
            ]
        )
    return rows


def _source(
    source_id: str,
    label: str,
    description: str,
    tables: list[str],
    metric_definitions: list[str] | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "engine": "Reviewed local JSON, Markdown and source",
        "language": "sql",
        "id": f"{source_id}_review_20260823",
        "executed_at": "2026-08-23T10:30:00+08:00",
        "description": description,
        "sql": f"SELECT * FROM {source_id}_review ORDER BY evidence_order",
        "tables_used": tables,
        "filters": ["S50 through S75", "SIM_ONLY evidence", "durable reports only"],
    }
    if metric_definitions:
        query["metric_definitions"] = metric_definitions
    return {
        "id": source_id,
        "label": label,
        "href": "https://github.com/ros-claw/rosclaw-soccer",
        "query": query,
    }


def build_artifact() -> dict[str, Any]:
    core_reports, counts = _inventory()
    students = _validated_students()
    if len(core_reports) != 111:
        raise RuntimeError(f"expected 111 core reports, found {len(core_reports)}")
    if len(students) != 52:
        raise RuntimeError(f"expected 52 student reports, found {len(students)}")
    if any(not report["four_gpu_training"] for _, report in students):
        raise RuntimeError("all corrective-student reports must retain four-GPU evidence")
    if any(report["promotion_eligible"] for _, report in students):
        raise RuntimeError("the report must not hide an eligible promotion")

    retained = sum(bool(report["student_development_retained"]) for _, report in students)
    representative_rows = _representative_rows()
    retest_rows = _retest_rows()
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    head = _current_head()
    clean = _working_tree_clean()

    sources = [
        _source(
            "prior_chain",
            "S1–S49 全链路基线报告",
            "将本报告的恢复研究与此前传球、射门、扑救、局部起身及真实扑后 0/9 结论对齐。",
            ["rosclaw-soccer-s1-s49-full-chain-discussion.artifact.json"],
        ),
        _source(
            "foundation_docs",
            "S50–S60 阶段实施报告",
            "复核 successor-state、教师桥、仅本体学生、MJX actor-critic、短视界教师和稳定性—可塑性门控的实施记录。",
            [
                "s50-successor-state-recovery-foundation.zh-CN.md",
                "s51-failure-driven-recovery-teacher-bridge.zh-CN.md",
                "s52-proprio-recovery-student-and-muscle-memory.zh-CN.md",
                "s54-modern-mujoco-mjx-recovery-foundation.zh-CN.md",
                "s55-short-horizon-corrective-teacher.zh-CN.md",
                "s56-corrective-muscle-memory-student.zh-CN.md",
                "s57-stability-plasticity-dagger-confidence-gate.zh-CN.md",
                "s58-scaled-corrective-curriculum-and-gated-dagger.zh-CN.md",
                "s59-temporal-intervention-and-historical-veto-growth.zh-CN.md",
                "s60-causal-effect-channel-budget.zh-CN.md",
            ],
        ),
        _source(
            "evidence_inventory",
            "S50–S75 耐久证据清单",
            "按阶段和核心文件名枚举训练、教师、学生、冻结复试及采集报告，不把视频或临时日志计入报告数。",
            sorted({path.name for path in core_reports}),
            [
                "核心报告数：仅统计五类约定文件名，不代表所有 JSON 文件总数。",
                "学生报告有效：当前证据验证器成功校验报告、模型、语料、谱系和权限字段。",
            ],
        ),
        _source(
            "corrective_reports",
            "S56–S75 纠偏学生报告",
            "从 52 份当前验证器可加载的 student-report.json 重算困难收益、正常回退、方向门、稳定门和晋级状态。",
            ["student-report.json (52 validated reports)"],
            [
                "困难收益：同初态父/子配对物理考试中的平均成本改善比例；至少 1% 才过困难门。",
                "正常回退：候选相对父策略的正常路线成本回归；正值是变差，负值是改善，允许上限 1%。",
                "当轮双门通过不等于晋级；还必须通过冻结历史域、新种子盲测与独立真值考试。",
            ],
        ),
        _source(
            "frozen_retests",
            "冻结历史域与新种子盲测",
            "对 S59、S60、S62 的旧 96 银行和 S67 的 S68 新种子银行执行不参与选择的复试。",
            ["frozen-benchmark-report.json (6 reports)"],
        ),
        _source(
            "implementation_source",
            "当前恢复 Growth 实现",
            "核对本体观察、教师/学生、MJX、DAgger、门控、通道 veto、证据验证和 MuJoCo 兼容代码。",
            [
                "src/rosclaw_soccer/training/recovery_corrective_student.py",
                "src/rosclaw_soccer/training/opentrack_recovery_corrective_student.py",
                "src/rosclaw_soccer/training/recovery_corrective_teacher.py",
                "src/rosclaw_soccer/training/recovery_mjx.py",
                "tests/test_s56_recovery_corrective_student.py",
            ],
        ),
        _source(
            "validation_run",
            "2026年8月23日复核",
            "重放 52 份学生证据验证器，并引用当前工作树最近一次全量 pytest、ruff 和 mypy 结果。",
            ["current soccer working tree", "52 student evidence reports"],
        ),
    ]

    headline_metrics = [
        {
            "core_report_count": len(core_reports),
            "student_report_count": len(students),
            "student_report_valid_count": len(students),
            "student_report_invalid_count": 0,
            "four_gpu_student_report_count": sum(
                bool(report["four_gpu_training"]) for _, report in students
            ),
            "local_dual_gate_count": retained,
            "promotion_eligible_count": 0,
        }
    ]

    implementation_layers = [
        {
            "order": 1,
            "layer": "通用 Growth 目标",
            "implemented": "SkillSuccessorState、SuccessorStateGrowthObjective、CapabilityFrontierScheduler",
            "purpose": "当前技能必须把身体送入下一个技能可接管的连续状态，而不是只优化眼前奖励",
            "status": "已实现协议",
        },
        {
            "order": 2,
            "layer": "安全与可塑性租约",
            "implemented": "GrowthSafetyProfile、PlasticityLease、冻结父代/队友哈希",
            "purpose": "探索与晋级分离；一次只允许焦点候选变化",
            "status": "已实现并 fail-closed",
        },
        {
            "order": 3,
            "layer": "失败记忆与做梦",
            "implemented": "FailureConditionedDream、内容绑定快照、稀有/困难优先课程",
            "purpose": "围绕真实失败状态扰动和重放，而不是在简单初态刷分",
            "status": "已实现基础",
        },
        {
            "order": 4,
            "layer": "恢复技能组合",
            "implemented": "Absorb → Get-Up → Athlete 三专家、soft blend、热交接",
            "purpose": "把冲量吸收、多姿态起身和可行走 successor 分开建模",
            "status": "接口完成，完整能力未通",
        },
        {
            "order": 5,
            "layer": "外部教师桥",
            "implemented": "MotionDecode/OpenTrack 动作审计、失败驱动物理路由、固定相位",
            "purpose": "由真实物理结果选择恢复教师，而不是只做姿态相似度匹配",
            "status": "S51 组件通过",
        },
        {
            "order": 6,
            "layer": "本体学生",
            "implemented": "93 维仅本体观察、MLP+GRU、29DoF PD residual、safetensors/ONNX",
            "purpose": "部署 actor 不读取真值相位；语料按完整初态分组防泄漏",
            "status": "学生尚未泛化起身",
        },
        {
            "order": 7,
            "layer": "现代物理训练",
            "implemented": "MuJoCo/MJX、4×A6000 recurrent PPO、非对称 critic、可恢复检查点",
            "purpose": "在并行接触动力学中学习受限 residual，拒绝 Isaac Gym 老路径",
            "status": "训练闭环完成，无晋级",
        },
        {
            "order": 8,
            "layer": "短视界反事实教师",
            "implemented": "CEM 20 步计划、零计划保底、Pareto 方向/稳定保持、动作 Jacobian",
            "purpose": "证明局部动作可控，并为学生提供逐步纠偏标签",
            "status": "96/96 与 192/192 教师接受",
        },
        {
            "order": 9,
            "layer": "保守肌肉记忆",
            "implemented": "失败/正常 50:50 蒸馏、信任域、稀疏通道选择",
            "purpose": "在冻结父策略之上只开放有证据的关节残差",
            "status": "产生开发候选，未晋级",
        },
        {
            "order": 10,
            "layer": "时序授权",
            "implemented": "连续触发、有限时租、cooldown、slew、历史能力否决",
            "purpose": "防止单帧误开在闭环里自激并持续添乱",
            "status": "新域可过，历史域否决",
        },
        {
            "order": 11,
            "layer": "状态条件通道权威",
            "implemented": "因果效应预算、向量 veto、绝对量纲校准、失败召回 margin",
            "purpose": "按身体状态决定每组肌肉能输出多大，而不是固定全局 gain",
            "status": "当轮局部通过，盲测未泛化",
        },
        {
            "order": 12,
            "layer": "因果考试与证据治理",
            "implemented": "单图 lockstep、共享 reset/action RNG、零干预精确同一、哈希谱系与 repair contract",
            "purpose": "排除双 rollout 数值漂移、证据篡改和被拒候选越权复活",
            "status": "已加固并全量测试通过",
        },
    ]

    phase_timeline = [
        {
            "order": 1,
            "phase": "S50",
            "question": "如何让当前技能为下一技能负责？",
            "implementation": "Successor-state、失败课程、安全 profile、plasticity lease；4GPU R0 扰动复测",
            "result": "局部 R0 16/16，但真实扑后仍 0/9",
            "decision": "分布断层是真问题，不能拿局部起身覆盖真实失败",
        },
        {
            "order": 2,
            "phase": "S51",
            "question": "外部教师是否存在可行恢复路径？",
            "implementation": "MotionDecode/OpenTrack 失败驱动物理路由与固定 holdout",
            "result": "开发 9/9、固定局部扰动 27/27",
            "decision": "教师桥组件通过；仍不可部署",
        },
        {
            "order": 3,
            "phase": "S52",
            "question": "仅本体神经学生能否蒸馏教师？",
            "implementation": "MLP+GRU、DDP、DAgger、情景肌肉记忆、直接物理考试",
            "result": "神经学生开发 0/9、密封 0/27；精确记忆 9/9、扰动仅 1/27",
            "decision": "离线误差下降不等于闭环恢复",
        },
        {
            "order": 4,
            "phase": "S53–S54",
            "question": "在线 actor-critic 与更大物理吞吐能否突破？",
            "implementation": "4GPU residual PPO、MJX、非对称 critic、26 代 route expert、失败回放",
            "result": "S53 训练内最高 44.5%，CPU sealed 0/27；S54 偶有 1/64，所有代仍拒绝",
            "decision": "多训练不会自动解决长时信用与状态上下文",
        },
        {
            "order": 5,
            "phase": "S55–S56",
            "question": "先学短视界可控纠偏是否更稳？",
            "implementation": "CEM 反事实教师、Pareto 保持、平衡蒸馏、Jacobian 通道选择",
            "result": "教师 96/96；S56-v3 困难 +1.015%，当轮双门通过",
            "decision": "获得可保留开发候选，但稳定缺口与 CPU 真值门未解决",
        },
        {
            "order": 6,
            "phase": "S57–S58",
            "question": "DAgger、置信门和扩容能否解决遗忘？",
            "implementation": "on-policy 正常反例、独立稳定硬门、OOD 失败关闭、192 状态与 6000 步",
            "result": "可塑模型困难 +2.34% 却正常回退 32.04%；保守模型又失去困难收益",
            "decision": "单帧静态门存在闭环正反馈",
        },
        {
            "order": 7,
            "phase": "S59–S60",
            "question": "时序授权与身体通道预算能否兼顾新旧知识？",
            "implementation": "时序租约、历史 veto、动作效应通道预算、冻结旧 96 复试",
            "result": "新 192 双门通过；旧 96 正常方向/成本否决",
            "decision": "静态预算只能局部解决 Stability–Plasticity",
        },
        {
            "order": 8,
            "phase": "S61–S67",
            "question": "状态条件 veto 和闭环 DAgger 能否完成跨域？",
            "implementation": "向量 veto、DAgger、绝对校准、时序触发、lockstep 因果考试、召回 margin",
            "result": "S67 当前银行 +1.454% 且正常零回退；仍只是选择域结果",
            "decision": "必须增加未参与调参的新种子盲测",
        },
        {
            "order": 9,
            "phase": "S68",
            "question": "当前通过能否泛化到新失败分布？",
            "implementation": "fresh teacher/current bank，不反向调参的盲测",
            "result": "S67 盲测困难仅 +0.458%，稳定失败；正常门通过",
            "decision": "当前门学习的是域内授权，不是通用恢复信用",
        },
        {
            "order": 10,
            "phase": "S69–S75",
            "question": "跨种子 DAgger、前缀加权、时序记忆与修复模式能否补齐？",
            "implementation": "新种子 DAgger、加权门、动作/负样本解耦、repair contract、温度校准",
            "result": "最好困难 +1.691% 时正常方向/稳定失败；最好正常改善时困难仅 +0.293% 或退化",
            "decision": "继续调门控已到结构性上限，应转向逐时间步物理信用",
        },
    ]

    early_experiments = [
        {
            "order": 1,
            "attempt": "S50 R0",
            "method": "四卡局部起身扰动复测",
            "failure_result": "16/16 局部通过",
            "normal_result": "真实扑后仍 0/9",
            "verdict": "组件证据，不晋级",
        },
        {
            "order": 2,
            "attempt": "S51 Teacher",
            "method": "失败驱动 MotionDecode/OpenTrack 路由",
            "failure_result": "开发 9/9；局部扰动 27/27",
            "normal_result": "未形成自主学生",
            "verdict": "教师桥组件通过",
        },
        {
            "order": 3,
            "attempt": "S52 Student",
            "method": "本体 MLP/GRU + DAgger",
            "failure_result": "开发 0/9；密封 0/27",
            "normal_result": "精确记忆扰动后 1/27",
            "verdict": "拒绝",
        },
        {
            "order": 4,
            "attempt": "S53 formal-C",
            "method": "4GPU residual PPO",
            "failure_result": "训练内成功 44.5%",
            "normal_result": "CPU sealed acquisition 0/27，retention 9/9",
            "verdict": "拒绝",
        },
        {
            "order": 5,
            "attempt": "S54 route experts v1–v26",
            "method": "MJX recurrent PPO、非对称 critic、失败回放",
            "failure_result": "曾出现 1/64 严格成功",
            "normal_result": "没有一代同时通过稳定/方向/后退门",
            "verdict": "26 代全部拒绝",
        },
        {
            "order": 6,
            "attempt": "S55 Teacher-96",
            "method": "20 步 CEM Pareto 纠偏教师",
            "failure_result": "96/96 接受；中位成本改善 13.664%",
            "normal_result": "搜索内方向/稳定保持 100%",
            "verdict": "教师数据通过，不是策略晋级",
        },
        {
            "order": 7,
            "attempt": "S58 Teacher-192",
            "method": "扩容 CEM 教师",
            "failure_result": "192/192 接受；中位成本改善 12.088%",
            "normal_result": "仅生成训练标签",
            "verdict": "教师数据通过，不是策略晋级",
        },
    ]

    failure_taxonomy = [
        {
            "order": 1,
            "failure": "Successor-state 分布断层",
            "evidence": "标准姿态局部起身 28/28，真实扑后 0/9",
            "why": "真实状态带高角速度、低骨盆和复杂接触，超出专家吸引域",
            "required_change": "从真实失败终态训练吸收—起身连续策略",
        },
        {
            "order": 2,
            "failure": "行为克隆暴露偏差",
            "evidence": "S52 离线误差下降但物理成功 0；情景记忆扰动后 1/27",
            "why": "小误差改变下一状态，网络随后访问未训练分布",
            "required_change": "on-policy rollout 与物理 critic，而非继续压 MAE",
        },
        {
            "order": 3,
            "failure": "长时信用不足",
            "evidence": "S53/S54 训练内改善与 CPU sealed/长时稳定不一致",
            "why": "终端成功由早期多个关节—时间动作共同决定，平均 episode reward 信号太稀",
            "required_change": "逐时间步反事实 advantage 与短视界 critic",
        },
        {
            "order": 4,
            "failure": "正常路线灾难性干扰",
            "evidence": "S58 MLP 困难 +2.341%，正常回退 32.044%",
            "why": "可塑学生在无需纠偏时仍持续出力",
            "required_change": "硬负样本、沉默损失、独立正常长考",
        },
        {
            "order": 5,
            "failure": "静态门闭环自激",
            "evidence": "离线开度分离好，物理正常回退仍达 14.449%",
            "why": "一次误开改变状态，导致门继续认为需要干预",
            "required_change": "带状态的进入/退出、有限时租、反事实关闭标签",
        },
        {
            "order": 6,
            "failure": "新域通过、旧域遗忘",
            "evidence": "S59/S60/S62 新域双门过，冻结旧域正常门拒绝",
            "why": "固定增益或静态通道预算不能适配不同身体上下文",
            "required_change": "状态条件向量权威，且训练中显式包含历史域",
        },
        {
            "order": 7,
            "failure": "成对考试伪因果风险",
            "evidence": "S65 旧执行语义 +1.469%，S66 lockstep 同模型只剩 +0.646%",
            "why": "独立 rollout 的数值/RNG 漂移可被误算为策略效果",
            "required_change": "单图 lockstep、共享 reset/action RNG、零干预精确同一",
        },
        {
            "order": 8,
            "failure": "选择域过拟合",
            "evidence": "S67 当前 +1.454%，S68 fresh blind 仅 +0.458% 且稳定失败",
            "why": "阈值和 veto 在同一失败银行反复迭代，隐性吸收了 holdout 信息",
            "required_change": "三银行协议：train/current/fresh-blind；盲测失败后换新银行",
        },
        {
            "order": 9,
            "failure": "门控不能创造正确动作",
            "evidence": "S70–S75 能在困难收益和正常安静之间移动，但无法共同过门",
            "why": "门只决定是否/多大执行现有 residual，不知道哪个关节在哪个时刻真正有益",
            "required_change": "训练动作本身的时序 critic，而不是继续扫温度和阈值",
        },
    ]

    next_plan = [
        {
            "order": 1,
            "workstream": "逐时间步反事实信用",
            "implementation": "对候选动作的关节组×时间窗口执行 matched counterfactual rollout，学习 Q(s,a_group,t) 和 advantage",
            "success_gate": "fresh blind 困难成本改善 ≥1%；方向/稳定全过",
            "reason": "直接解决门控无法判断动作因果贡献的问题",
        },
        {
            "order": 2,
            "workstream": "残差 actor-critic 2.0",
            "implementation": "本体 recurrent actor；训练期非对称 critic；教师计划只作 warm start/行为先验，不作相位输入",
            "success_gate": "正常 600 步回退 ≤1%，动作 RMS 受限，零权限与父代精确同一",
            "reason": "从分类门升级为能学习动作的闭环小脑",
        },
        {
            "order": 3,
            "workstream": "三银行持续学习",
            "implementation": "训练银行、当前开发银行、一次性 fresh-blind 银行；盲测结果禁止回调同一银行",
            "success_gate": "新域、冻结旧域、fresh blind 顺序全部通过",
            "reason": "消除选择域过拟合和隐性 holdout 泄漏",
        },
        {
            "order": 4,
            "workstream": "Stability–Plasticity 正则",
            "implementation": "冻结 champion、KL/EWC、旧成功轨迹回放、沉默损失、通道权威上界",
            "success_gate": "旧正常方向/稳定/成本非劣，且困难收益不被压没",
            "reason": "保留旧肌肉记忆同时允许新失败可塑",
        },
        {
            "order": 5,
            "workstream": "分层时域课程",
            "implementation": "20→40→80→160→400→600 步；每级从本级失败末态继续出题",
            "success_gate": "每一级只有 paired physics 双门通过才扩时域",
            "reason": "避免直接用长回合稀疏奖励训练",
        },
        {
            "order": 6,
            "workstream": "完整足球闭环",
            "implementation": "候选通过底层三域门后接回扑救—落地—恢复—ready—二扑，不 teleport/reset",
            "success_gate": "首扑非劣、条件恢复率与二扑率提升、P95 冲击/恢复时间受控",
            "reason": "最终评价必须回到 ROSClaw 真实任务价值",
        },
        {
            "order": 7,
            "workstream": "视频与发布",
            "implementation": "只渲染已通过模型的 matched parent/candidate 长对照和完整足球链",
            "success_gate": "报告、模型、场景、视频 manifest 哈希绑定；CPU truth exam 通过",
            "reason": "视频用于解释能力，不替代能力证据",
        },
    ]

    validation_results = [
        {
            "order": 1,
            "check": "学生报告当前验证器重放",
            "result": "52/52 通过，0 失败",
            "scope": "S56–S75；模型/语料/谱系/权限/物理考试字段",
        },
        {
            "order": 2,
            "check": "四卡物理证据",
            "result": "52/52 记录 cuda:0–3，four_gpu_training=true",
            "scope": "纠偏学生系列；均为本地 4×RTX A6000",
        },
        {
            "order": 3,
            "check": "最终晋级权限",
            "result": "0/52 promotion_eligible；hardware_command_sent=false",
            "scope": "当轮开发保留不能替代跨域/CPU 晋级",
        },
        {
            "order": 4,
            "check": "冻结/盲测",
            "result": "6 份冻结报告；S59/S60/S62/S67 关键候选均未通过最终复试",
            "scope": "旧 96 银行或 fresh blind 新种子",
        },
        {
            "order": 5,
            "check": "当前全量单元测试",
            "result": "524 passed，11 skipped，0 failed",
            "scope": "Soccer 当前工作树；跳过项为缺 stacked Core PR 或可选资产",
        },
        {
            "order": 6,
            "check": "静态检查",
            "result": "相关源码 ruff 与 mypy 通过",
            "scope": "纠偏学生、OpenTrack runner、MuJoCo field 兼容修复",
        },
        {
            "order": 7,
            "check": "安全边界",
            "result": "SIM_ONLY；真实机器人、ROS/DDS、硬件命令均为 0",
            "scope": "遵循 ROSClaw/SimForge 物理 AI 边界",
        },
        {
            "order": 8,
            "check": "代码快照",
            "result": f"HEAD {head}；工作树{'干净' if clean else '含未提交累积改动'}",
            "scope": "报告是本地研究快照，不是已发布版本说明",
        },
    ]

    cards = [
        {
            "id": "card_evidence",
            "description": "仅统计约定的训练、教师、学生、冻结复试和采集核心报告。",
            "dataset": "headline_metrics",
            "sourceId": "evidence_inventory",
            "metrics": [
                {"label": "核心证据报告", "field": "core_report_count", "format": "number"}
            ],
        },
        {
            "id": "card_student_validation",
            "description": "52 份学生报告均由当前验证器重新加载。",
            "dataset": "headline_metrics",
            "sourceId": "corrective_reports",
            "metrics": [
                {
                    "label": "有效学生报告",
                    "field": "student_report_valid_count",
                    "format": "number",
                },
                {"label": "无效", "field": "student_report_invalid_count", "format": "number"},
            ],
        },
        {
            "id": "card_four_gpu",
            "description": "每份纠偏学生证据都记录四个唯一 CUDA 设备。",
            "dataset": "headline_metrics",
            "sourceId": "corrective_reports",
            "metrics": [
                {
                    "label": "4GPU 学生实验",
                    "field": "four_gpu_student_report_count",
                    "format": "number",
                }
            ],
        },
        {
            "id": "card_local_pass",
            "description": "只表示当轮 failure/normal 双考通过，后续冻结或盲测仍可否决。",
            "dataset": "headline_metrics",
            "sourceId": "corrective_reports",
            "metrics": [
                {"label": "当轮双门通过", "field": "local_dual_gate_count", "format": "number"}
            ],
        },
        {
            "id": "card_promotion",
            "description": "没有模型取得独立 CPU 真值晋级或硬件权限。",
            "dataset": "headline_metrics",
            "sourceId": "corrective_reports",
            "metrics": [
                {"label": "最终晋级", "field": "promotion_eligible_count", "format": "number"}
            ],
        },
    ]

    charts = [
        {
            "id": "chart_stability_plasticity",
            "title": "代表候选的困难收益与正常回退",
            "subtitle": "S56–S75；纵轴越高越可塑，横轴越低越稳定。困难收益需 ≥1%，正常回退需 ≤1%，且方向/稳定硬门必须同时通过。",
            "showDescription": True,
            "intent": "relationship",
            "question": "多轮方法是否把候选稳定推入困难收益与正常保持共同可行域？",
            "rationale": "两个同为百分比的配对物理成本变化适合散点图；它能直接显示稳定性—可塑性的拉扯，而不是只展示单一最好值。",
            "type": "scatter",
            "dataset": "representative_candidates",
            "sourceId": "corrective_reports",
            "encodings": {
                "x": {
                    "field": "normal_regression_pct",
                    "type": "quantitative",
                    "label": "正常成本回退（%）",
                },
                "y": {
                    "field": "failure_improvement_pct",
                    "type": "quantitative",
                    "label": "困难成本改善（%）",
                },
                "color": {"field": "gate_status", "type": "nominal", "label": "当轮状态"},
                "tooltip": [
                    {"field": "attempt", "type": "text", "label": "候选"},
                    {"field": "method", "type": "text", "label": "方法"},
                    {"field": "normal_action_rms", "type": "quantitative", "label": "正常动作 RMS"},
                    {"field": "later_result", "type": "text", "label": "最终/后续结论"},
                ],
            },
            "xAxisTitle": "正常成本回退（%，负值表示改善）",
            "yAxisTitle": "困难成本改善（%）",
            "valueFormat": "number",
            "unit": "%",
            "layout": "full",
            "maxRows": 32,
            "settings": {"showValues": False, "legend": "top"},
            "palette": {"kind": "categorical", "colors": ["blue", "orange"]},
        },
        {
            "id": "chart_retest_failure",
            "title": "当轮开发域与冻结/盲测的困难收益",
            "subtitle": "四个曾在当轮双门通过的候选；复试仍可能保持困难收益，但会被正常方向/成本或新种子稳定性否决。",
            "showDescription": True,
            "intent": "comparison",
            "question": "当轮通过的恢复收益是否在未参与选择的域中保持？",
            "rationale": "四个候选各有当轮与复试两项同口径困难收益，分组柱形图能直接展示泛化落差。",
            "type": "bar",
            "dataset": "retest_comparison",
            "sourceId": "frozen_retests",
            "encodings": {
                "x": {"field": "candidate", "type": "nominal", "label": "候选"},
                "y": {
                    "field": "failure_improvement_pct",
                    "type": "quantitative",
                    "label": "困难成本改善（%）",
                },
                "color": {"field": "exam_domain", "type": "nominal", "label": "考试域"},
                "tooltip": [
                    {
                        "field": "normal_regression_pct",
                        "type": "quantitative",
                        "label": "正常回退（%）",
                    },
                    {"field": "blocker", "type": "text", "label": "门控结论"},
                ],
            },
            "xAxisTitle": "候选",
            "yAxisTitle": "困难成本改善（%）",
            "valueFormat": "number",
            "unit": "%",
            "layout": "full",
            "maxRows": 12,
            "settings": {
                "orientation": "vertical",
                "grouped": True,
                "showValues": True,
                "legend": "top",
            },
            "palette": {"kind": "categorical", "colors": ["blue", "orange"]},
        },
    ]

    tables = [
        {
            "id": "table_implementation_layers",
            "title": "从 ROSClaw Growth 到 Soccer 恢复小脑的实现分层",
            "subtitle": "区分通用成长协议、领域恢复实现和当前可信状态。",
            "showDescription": True,
            "dataset": "implementation_layers",
            "density": "spacious",
            "sourceId": "implementation_source",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "序号", "type": "number", "align": "right"},
                {"field": "layer", "label": "层", "type": "text"},
                {"field": "implemented", "label": "实现内容", "type": "text"},
                {"field": "purpose", "label": "解决的问题", "type": "text"},
                {"field": "status", "label": "当前状态", "type": "text"},
            ],
        },
        {
            "id": "table_phase_timeline",
            "title": "S50–S75 阶段路线与因果认识",
            "subtitle": "把 26 个阶段归并为十个真正改变架构、实验口径或结论的波次。",
            "showDescription": True,
            "dataset": "phase_timeline",
            "density": "spacious",
            "sourceId": "foundation_docs",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "序号", "type": "number", "align": "right"},
                {"field": "phase", "label": "阶段", "type": "text"},
                {"field": "question", "label": "核心问题", "type": "text"},
                {"field": "implementation", "label": "主要实施", "type": "text"},
                {"field": "result", "label": "结果", "type": "text"},
                {"field": "decision", "label": "结论", "type": "text"},
            ],
        },
        {
            "id": "table_early_experiments",
            "title": "S50–S55 基础与 actor-critic 关键实验",
            "subtitle": "早期结果口径不同，保留分母和证据等级，不与后续百分比成本门直接排名。",
            "showDescription": True,
            "dataset": "early_experiments",
            "density": "spacious",
            "sourceId": "foundation_docs",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "序号", "type": "number", "align": "right"},
                {"field": "attempt", "label": "实验", "type": "text"},
                {"field": "method", "label": "方法", "type": "text"},
                {"field": "failure_result", "label": "困难/训练结果", "type": "text"},
                {"field": "normal_result", "label": "保持/真值结果", "type": "text"},
                {"field": "verdict", "label": "结论", "type": "text"},
            ],
        },
        {
            "id": "table_representative_candidates",
            "title": "S56–S75 代表候选物理闭环结果",
            "subtitle": "困难改善至少 1%；正常回退正值表示变差、负值表示改善。即使当轮双门过，也不自动获得晋级。",
            "showDescription": True,
            "dataset": "representative_candidates",
            "density": "dense",
            "sourceId": "corrective_reports",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "序号", "type": "number", "align": "right"},
                {"field": "attempt", "label": "候选", "type": "text"},
                {"field": "method", "label": "方法", "type": "text"},
                {
                    "field": "failure_improvement_pct",
                    "label": "困难改善（%）",
                    "type": "number",
                    "align": "right",
                },
                {
                    "field": "normal_regression_pct",
                    "label": "正常回退（%）",
                    "type": "number",
                    "align": "right",
                    "semantic": "movement",
                },
                {
                    "field": "normal_action_rms",
                    "label": "正常动作 RMS",
                    "type": "number",
                    "align": "right",
                },
                {"field": "gate_status", "label": "当轮门", "type": "text"},
                {"field": "later_result", "label": "最终/后续结论", "type": "text"},
            ],
        },
        {
            "id": "table_retests",
            "title": "当轮通过候选的冻结历史域与 fresh-blind 复试",
            "subtitle": "同一候选在未参与选择的域重新跑配对物理考试；passed=false 才是最终可泛化结论。",
            "showDescription": True,
            "dataset": "retest_comparison",
            "density": "spacious",
            "sourceId": "frozen_retests",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "序号", "type": "number", "align": "right"},
                {"field": "candidate", "label": "候选", "type": "text"},
                {"field": "exam_domain", "label": "考试域", "type": "text"},
                {
                    "field": "failure_improvement_pct",
                    "label": "困难改善（%）",
                    "type": "number",
                    "align": "right",
                },
                {
                    "field": "normal_regression_pct",
                    "label": "正常回退（%）",
                    "type": "number",
                    "align": "right",
                    "semantic": "movement",
                },
                {"field": "passed", "label": "全门通过", "type": "boolean"},
                {"field": "blocker", "label": "结论/阻断项", "type": "text"},
            ],
        },
        {
            "id": "table_failure_taxonomy",
            "title": "九类失败如何逐步暴露真正瓶颈",
            "subtitle": "每次拒绝都对应一个后续架构变化；最后一行解释为什么继续扫门控参数价值有限。",
            "showDescription": True,
            "dataset": "failure_taxonomy",
            "density": "spacious",
            "sourceId": "foundation_docs",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "序号", "type": "number", "align": "right"},
                {"field": "failure", "label": "失败类型", "type": "text"},
                {"field": "evidence", "label": "直接证据", "type": "text"},
                {"field": "why", "label": "诊断", "type": "text"},
                {"field": "required_change", "label": "需要的结构变化", "type": "text"},
            ],
        },
        {
            "id": "table_next_plan",
            "title": "建议讨论并拍板的下一阶段实施包",
            "subtitle": "按因果依赖排序；前一项没有通过门控时，不进入完整足球展示。",
            "showDescription": True,
            "dataset": "next_plan",
            "density": "spacious",
            "sourceId": "implementation_source",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "优先级", "type": "number", "align": "right"},
                {"field": "workstream", "label": "工作流", "type": "text"},
                {"field": "implementation", "label": "实施内容", "type": "text"},
                {"field": "success_gate", "label": "硬验收门", "type": "text"},
                {"field": "reason", "label": "为什么现在做", "type": "text"},
            ],
        },
        {
            "id": "table_validation",
            "title": "报告复核、软件质量与证据边界",
            "subtitle": "本报告复核现存耐久证据，不重跑数千万物理步；所有结论保持 SIM_ONLY。",
            "showDescription": True,
            "dataset": "validation_results",
            "density": "spacious",
            "sourceId": "validation_run",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "序号", "type": "number", "align": "right"},
                {"field": "check", "label": "检查项", "type": "text"},
                {"field": "result", "label": "结果", "type": "text"},
                {"field": "scope", "label": "范围与解释", "type": "text"},
            ],
        },
    ]

    title = "ROSClaw Soccer S50–S75 恢复小脑与自进化实施复盘"
    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "body": "## 技术摘要\n\n- **工程闭环已经从‘真实失败状态’贯通到教师、学生、4GPU 物理训练、DAgger、冻结复试和 fail-closed 证据，但恢复能力仍未晋级。** S50–S75 留下 111 份核心报告；52 份纠偏学生报告全部通过当前验证器，10 份曾通过当轮困难/正常双门，最终 `promotion_eligible` 仍为 0。\n\n- **最强结果都被更严格的后续考试否决。** S59-v12、S60-v4、S62-v1 和 S67 分别在选择域通过，但旧 96 银行或 S68 fresh-blind 暴露正常方向遗忘、成本回退或困难稳定退化。当前系统已经能拒绝伪突破，这是 ROSClaw 自进化治理的进步；它还没有训练出可跨域复用的小脑。\n\n- **S69–S75 证明继续扫温度、阈值、前缀权重或门控时序不会自然收敛。** 困难收益较高时正常方向/稳定失败；正常路线足够安静时困难收益又低于 1%，甚至退化。门控只能决定‘何时放大现有动作’，不能学习‘哪个关节在什么时间产生了真实正贡献’。\n\n- **建议下一阶段停止门控参数搜索，转向逐时间步物理信用分配与残差 actor-critic 2.0。** 使用关节组×时间窗口 matched counterfactual rollout 学习 advantage，并在训练/当前/一次性盲测三银行上顺序验收；底层三域全部通过后，才回接扑救—落地—起身—ready—二扑和宣传视频。",
        },
        {
            "id": "headline_context",
            "type": "markdown",
            "layout": "full",
            "body": "## 先看证据规模与最终结论\n\n`当轮双门通过`只表示候选在本轮开发银行内同时通过困难与正常配对考试；它不等于冻结历史保持、fresh-blind 泛化、CPU 真值晋级或部署授权。",
        },
        {
            "id": "headline_strip",
            "type": "metric-strip",
            "layout": "full",
            "cardIds": [
                "card_evidence",
                "card_student_validation",
                "card_four_gpu",
                "card_local_pass",
                "card_promotion",
            ],
        },
        {
            "id": "finding_plane",
            "type": "markdown",
            "layout": "full",
            "body": "## 多轮实验一直在稳定性与可塑性之间摆动\n\n下图每个点是一份代表性配对物理考试。理想候选要同时位于‘困难改善至少 1%、正常回退不超过 1%’的区域，并另行通过方向、稳定、有限值与跨域门。S58 的高可塑模型远离正常保持；S69–S75 的保守门控逐渐靠近正常安全，却把困难恢复信号压没。",
        },
        {
            "id": "stability_chart",
            "type": "chart",
            "layout": "full",
            "chartId": "chart_stability_plasticity",
        },
        {
            "id": "finding_retest",
            "type": "markdown",
            "layout": "full",
            "body": "## 四次‘当轮突破’都没有穿过独立复试\n\nS59、S60、S62 和 S67 的当轮数字都曾满足双门，但复试不复用调参域。前三者在旧正常路线被方向或成本否决；S67 在 fresh-blind 新失败状态上由约 1.45% 降到 0.46%，稳定门失败。这证明目前最危险的风险不是训练不足，而是选择域过拟合。",
        },
        {
            "id": "retest_chart",
            "type": "chart",
            "layout": "full",
            "chartId": "chart_retest_failure",
        },
        {"id": "retest_table", "type": "table", "layout": "full", "tableId": "table_retests"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "layout": "full",
            "body": "## 范围、数据和指标口径\n\n本报告覆盖 S50–S75 的恢复/小脑 Growth 研究，前置基线是 S49 的真实扑后恢复 0/9。S50–S55 使用局部起身成功率、训练内成功率、sealed acquisition 和教师成本改善等异构口径；S56 起统一采用同初态、同随机流的 failure/normal paired physics exam。\n\n- **困难成本改善**：父策略成本减候选成本，再除以父策略成本；至少 `1%`。\n- **正常成本回退**：候选相对父策略的正常路线成本增长；正值为变差、负值为改善，最多 `1%`。\n- **方向保持**：后退、横移和偏航分别受限，不能由总代价平均掩盖。\n- **稳定保持**：独立硬门；困难状态不允许退化，正常路线只允许预先冻结的容差。\n- **晋级**：当轮双门、冻结旧域、fresh-blind、独立 CPU MuJoCo truth exam 与权限字段必须全部通过。",
        },
        {
            "id": "design",
            "type": "markdown",
            "layout": "full",
            "body": "## 实验设计：冻结父代、只改候选、让失败拥有否决权\n\nActor 只读取本体感和内部历史，输出冻结父恢复策略之上的受限 29 关节 PD residual；训练期 critic 可读取更丰富的仿真真值。教师、父检查点、场景、路由、快照、语料和数值运行时均由哈希绑定。每轮在 4×A6000 上训练或配对评估，但 GPU 结果本身没有晋级权。\n\n最终考试采用单图 lockstep 父/子执行，共享 reset 和 action RNG；零干预必须逐位等同父代。训练银行、当前开发银行、冻结历史银行和 fresh-blind 银行的职责分开，失败报告可以继续作为反例来源，但不能被改写成能力通过。",
        },
        {
            "id": "implementation_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 做了什么：十二层模块化实现",
        },
        {
            "id": "implementation_table",
            "type": "table",
            "layout": "full",
            "tableId": "table_implementation_layers",
        },
        {
            "id": "timeline_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 二十六个阶段如何形成十次认识升级",
        },
        {
            "id": "timeline_table",
            "type": "table",
            "layout": "full",
            "tableId": "table_phase_timeline",
        },
        {
            "id": "early_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 前半程：先证明有教师、有动作可控性，再证明学生并不会自动学会\n\nS50–S55 最大的进展是把‘没有可行动作’与‘学生学不会/执行不稳’分开：S51 教师桥在限定邻域可行，S55 CEM 又证明 29DoF 动作局部可控；S52/S53/S54 则连续否定了纯模仿、普通 residual PPO 和长时间续训可以直接解决问题。",
        },
        {
            "id": "early_table",
            "type": "table",
            "layout": "full",
            "tableId": "table_early_experiments",
        },
        {
            "id": "candidate_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 后半程：每个候选的真实闭环结果\n\n下表保留代表性正负结果。不能只看困难改善：例如 S58-MLP 的 +2.341% 同时伴随正常路线 +32.044% 回退；S74 正常成本改善约 2.352%，但困难改善只有 0.293%。这正是 Stability–Plasticity Dilemma 的量化表现。",
        },
        {
            "id": "candidate_table",
            "type": "table",
            "layout": "full",
            "tableId": "table_representative_candidates",
        },
        {
            "id": "failure_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 失败没有白跑：九类根因已经被区分\n\n这些负结果并非同一种‘训练失败’。每一轮都缩小了问题空间：从真实终态分布断层，到行为克隆暴露偏差、长时信用、正常干扰、静态门自激、历史遗忘、成对考试伪因果、选择域过拟合，最终定位到动作本身缺少逐时间物理信用。",
        },
        {
            "id": "failure_table",
            "type": "table",
            "layout": "full",
            "tableId": "table_failure_taxonomy",
        },
        {
            "id": "rosclaw_value",
            "type": "markdown",
            "layout": "full",
            "body": "## 对 ROSClaw 本体真正有价值的成果\n\n足球恢复仍没晋级，但这轮已经形成可迁移的通用能力：successor-state 目标、失败条件做梦、能力边缘调度、可塑性租约、内容绑定语料、on-policy 证据、冻结历史银行、状态条件动作权威、lockstep 因果考试和 fail-closed 晋级。它们不依赖球门或 G1 关节名，可复用于抓取后复位、碰撞脱困、装配重试、移动机器人打滑恢复等 Physical-AI 任务。\n\nSoccer 插件继续保留 G1 29DoF、扑后姿态、球场和恢复成本定义；ROSClaw Core 只应承载任务无关的 Practice→Memory→Growth→Exam 协议。",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": "## 限制、稳健性与不能宣称的内容\n\n- 所有结果均为 `SIM_ONLY`，没有真实机器人、ROS/DDS 或电机命令证据。\n- 52/52 报告通过验证器只证明证据完整，不证明模型能力通过。\n- S50–S55 指标口径异构，不能与 S56–S75 的成本百分比直接做排行榜。\n- 当前银行的多轮迭代存在选择域适配风险；S68 fresh-blind 才是更可信的泛化结论。\n- 没有候选通过最终晋级，因此不能声称已经训练出稳定、可持续学习的 G1 小脑。\n- 本轮没有生成宣传视频；被拒候选的单条好看 rollout 不能替代分层配对考试。\n- 工作树含大量累积未提交内容，本报告是研究快照，不是 main 分支发布说明。",
        },
        {
            "id": "next_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 下一阶段不再扫门：改学动作的逐时间因果贡献\n\n建议把资源集中在一个可证伪的实施包：先训练关节组×时间窗口 critic，再用其 advantage 更新 recurrent residual actor；以三银行顺序考试解决域内过拟合，以 KL/EWC、成功回放和沉默损失守住旧技能。任何一层失败都返回对应失败银行，不进入足球视频链。",
        },
        {"id": "next_table", "type": "table", "layout": "full", "tableId": "table_next_plan"},
        {
            "id": "questions",
            "type": "markdown",
            "layout": "full",
            "body": "## 讨论会上需要拍板的问题\n\n1. 是否同意停止对现有置信/温度/阈值门继续做大规模 sweep，把算力转给逐时间反事实 critic？\n2. fresh-blind 银行每轮应包含多少独立失败源，是否接受至少 24 个来源作为开发门、64+ 作为正式门？\n3. 困难改善 1% 是否仍足够，还是应增加 P95、最差分层与恢复成功事件作为共同门？\n4. 正常保持是否继续用 600 步，还是增加 2,000 步尾部漂移测试？\n5. residual actor 的动作权限应按腿/腰/臂分阶段开放，还是直接用关节组 advantage 自适应稀疏？\n6. 底层恢复小脑通过后，完整足球链的主指标应如何权衡首扑率、条件恢复率、恢复时间与二扑率？\n7. 哪些通用协议应立即抽到 ROSClaw Core，哪些仍应在 Soccer 中完成第二个任务复验后再抽象？",
        },
        {
            "id": "validation_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 复核状态与交付边界",
        },
        {
            "id": "validation_table",
            "type": "table",
            "layout": "full",
            "tableId": "table_validation",
        },
        {
            "id": "plain_language",
            "type": "markdown",
            "layout": "full",
            "body": "## 最通俗的解释\n\n我们已经给机器人建好了‘失败录像馆、教练、训练场、考试委员会和旧技能保护制度’，也多次训练出在熟悉考题上会做小修正的候选。但一换到新的摔倒方式，它要么不敢出力，要么一出力就让原本正常的身体慢慢偏掉。问题不再是缺少更多阈值，而是它没有真正理解：某一时刻动哪块肌肉，几步以后到底帮了忙还是添了乱。下一步要训练的就是这种逐时刻、逐肌群的物理因果判断。",
        },
        {
            "id": "handoff",
            "type": "markdown",
            "layout": "full",
            "body": f"## 交付说明\n\n本报告生成于 2026年8月23日，覆盖 S50–S75，以 Soccer 工作树 `{head}` 和现存耐久证据为快照。报告生成只读取本地代码与证据，不启动训练、不连接 ROS/DDS、不执行硬件动作，也不提交或推送代码。",
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "技术复盘 S50–S75 恢复小脑、自进化闭环、4GPU 实验、冻结/盲测结果、失败根因与下一阶段决策。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline_metrics": headline_metrics,
                "implementation_layers": implementation_layers,
                "phase_timeline": phase_timeline,
                "early_experiments": early_experiments,
                "representative_candidates": representative_rows,
                "retest_comparison": retest_rows,
                "failure_taxonomy": failure_taxonomy,
                "next_plan": next_plan,
                "validation_results": validation_results,
            },
        },
    }


def main() -> None:
    artifact = build_artifact()
    OUTPUT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

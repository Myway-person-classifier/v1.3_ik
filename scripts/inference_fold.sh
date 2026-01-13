#!/usr/bin/env bash
set -euo pipefail

# Fold별 추론(검증/테스트)용 템플릿
# 주의: 현재 fold_trainer.py는 학습 + 검증 logits 저장을 수행합니다.
# 이미 학습된 모델로 테스트 로그잇만 생성하는 별도 스크립트는 구현되지 않았습니다.
# 사용자는 학습 시점에 --save_fold_logits True로 검증/테스트 로그잇을 확보하는 것을 권장합니다.

echo "현재 버전에서는 fold_trainer를 통해 학습 시 생성된 로그잇을 사용하세요."
echo "테스트 로그잇이 필요하면 각 Fold 학습 시 --is_submission True 지원이 필요하며, 이는 별도 구현이 요구됩니다."
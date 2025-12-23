from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path

# 1. 설정
OLD_RUN_DIR = "/datasets/v-geonhuison/JiT/baseline"         # 기존 event 파일들이 있는 디렉토리
NEW_RUN_DIR = "/datasets/v-geonhuison/JiT/test_1_n" # 새로 쓸 디렉토리

os.makedirs(NEW_RUN_DIR, exist_ok=True)

# 2. tag rename 규칙 정의
def rename_tag(tag: str) -> str:
    mapping = {
        "train_loss": "diff_loss",
        # 필요한 만큼 매핑 추가
    }
    # 매핑에 없으면 그대로 두고, 있으면 새 이름으로 변경
    return mapping.get(tag, tag)

# 3. 기존 event 읽기
ea = event_accumulator.EventAccumulator(OLD_RUN_DIR)
ea.Reload()

# 4. 새 writer 준비
writer = SummaryWriter(NEW_RUN_DIR)

# 5. scalar만 예시로 처리
scalar_tags = ea.Tags().get("scalars", [])

for old_tag in scalar_tags:
    new_tag = rename_tag(old_tag)
    scalar_events = ea.Scalars(old_tag)  # 각 요소가 (wall_time, step, value)

    for ev in scalar_events:
        # PyTorch SummaryWriter는 walltime 지정도 가능
        writer.add_scalar(
            new_tag,
            ev.value,
            global_step=ev.step,
            walltime=ev.wall_time,
        )

writer.close()

print("완료! 이제 `tensorboard --logdir logs_renamed` 로 보면 됨.")

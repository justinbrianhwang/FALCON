# FALCON 실험 실행 가이드 (실패 타입 확장)

실행에 필요한 내용만 담았습니다.

## 1. 환경 설치 (최초 1회 — 이미 falcon 환경이 있으면 건너뛰기)

```bash
conda env create -f environment.yml
conda activate falcon
pytest        # 전부 통과하면 환경 OK
```

이미 환경이 있는 경우에도 코드가 갱신되었으므로 `pytest`로 한 번 확인해 주세요.

## 2. 데이터 준비 (최초 1회 — 이미 mnist.pkl이 있으면 건너뛰기)

```bash
python scripts/prepare_data.py --datasets mnist
```

## 3. 실험 — 실패 타입 확장 (MNIST, 수십 분)

```bash
python experiments/run_failure_types.py
```

새 실패 타입 3종(라벨 오염, 과도한 클리핑, 저비트 양자화)을 MNIST에서 주입하고
스테이지 귀속 + 클라이언트 수준 localization까지 수행합니다.
`results/failure_types/summary.json`과 케이스별 리포트가 생성됩니다.

## 4. 결과 보내기

```bash
python scripts/collect_output.py
# -> tmp/Output_YYYY-MM-DD_HH-MM-SS.zip 생성됨
```

생성된 zip 파일을 보내주세요. `--full`은 따로 요청받은 경우에만 붙여주세요.

## 문제 생기면

- `pytest` 실패: `conda env remove -n falcon` 후 1번부터 재시도.
- 데이터 다운로드 실패: 네트워크 문제 — `--datasets mnist`로 재실행 (이미 받은 것은 건너뜀).
- 실험 중단: 같은 명령 재실행 후, 에러 메시지 전체와 함께 zip을 보내주세요.

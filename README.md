# GPT-6 Luna History Routing 실험 결과

## 1. 실험 목적

멀티턴 질의에서 현재 질문을 이해하기 위해 이전 대화 history가 필요한지 판단하고, 필요한 경우 어떤 turn만 선택하는 것이 효과적인지 비교했다.

비교한 방법은 다음 3가지다.

1. `always_all`
   - 항상 전체 history 사용

2. `binary_gate`
   - history 필요 여부만 `0/1`로 판단
   - `1`이면 전체 history 사용

3. `history_select`
   - 필요한 history turn만 직접 선택
   - 예: `1,3`

평가 데이터는 총 100개이며, history-dependent multi-turn 50개와 standalone single-turn 50개로 구성했다.

---

## 2. 전체 결과

| Method | Binary Acc | Binary F1 | Turn Precision | Turn Recall | Turn F1 | Exact Selection | Avg Selected Turns | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `always_all` | 0.500 | 0.667 | 0.359 | **1.000** | 0.528 | 0.080 | 6.77 | **0.000s** |
| `binary_gate` | 0.730 | 0.787 | 0.359 | **1.000** | 0.528 | 0.080 | 5.11 | 1.222s |
| `history_select` | **0.910** | **0.901** | **0.592** | 0.526 | **0.557** | **0.120** | **1.03** | 1.327s |

---

## 3. 핵심 결과

### 3.1 History가 필요한지 판단하는 것만 보면 `history_select`가 가장 좋음

Binary classification 기준으로:

```text
always_all      50.0%
binary_gate     73.0%
history_select  91.0%
```

`history_select`는 별도의 `0/1` 질문을 하지 않고도,

```text
0
```

또는

```text
1,3
```

처럼 turn을 직접 고르게 했는데 history 사용 여부 판단 정확도가 오히려 가장 높았다.

즉 이번 데이터에서는

```text
"history가 필요한가?"
```

만 따로 묻는 것보다

```text
"정확히 어떤 history가 필요한가?"
```

를 판단시키는 쪽이 모델에게 더 명확한 task였던 것으로 보인다.

`binary_gate`의 73% 대비 `history_select`는 91%로 **18%p 차이**가 난다.

---

## 4. `binary_gate`의 문제

`binary_gate` 결과:

```text
Binary Accuracy : 0.730
Binary F1       : 0.787
Avg turns       : 5.11
```

Binary gate 자체는 baseline보다 상당히 개선됐다.

하지만 구조가

```text
1 → history 전체 사용
0 → history 사용 안 함
```

이므로 history가 필요하다고 판단한 순간 관련 없는 turn까지 전부 들어간다.

그 결과 turn-level 성능은 `always_all`과 완전히 동일하다.

```text
                 Precision   Recall   F1
always_all        0.359      1.000   0.528
binary_gate       0.359      1.000   0.528
```

즉 binary routing은 **history를 넣을지 말지는 개선하지만, 어떤 history를 넣을지는 전혀 개선하지 못한다.**

평균 context도:

```text
always_all   6.77 turns
binary_gate  5.11 turns
```

으로 약 25% 정도밖에 줄지 않았다.

---

## 5. `history_select`의 가장 큰 장점: Context 압축

평균 선택 turn 수가 크게 감소했다.

```text
always_all      6.77
binary_gate     5.11
history_select  1.03
```

전체 history 대비:

```text
1.03 / 6.77 ≈ 15.2%
```

즉 `history_select`는 평균적으로 **약 85%의 history turn을 제거**했다.

Binary gate와 비교해도:

```text
1.03 / 5.11 ≈ 20.2%
```

약 **80% context reduction**이다.

LLM downstream inference 관점에서는 꽤 큰 차이다.

예를 들어 history가 평균 700 tokens라고 가정하면 대략:

```text
always_all      ~700 tokens
binary_gate     ~530 tokens
history_select  ~110 tokens
```

정도로 줄어들 가능성이 있다.

따라서 대화 history가 길어질수록 selection 방식의 비용 절감 효과는 더 커질 수 있다.

---

## 6. 하지만 `history_select`에도 중요한 문제점이 있음

가장 주의해서 볼 지표는:

```text
Turn Precision = 0.592
Turn Recall    = 0.526
```

이다.

Precision은 개선됐다.

```text
0.359 → 0.592
```

하지만 Recall은 크게 감소했다.

```text
1.000 → 0.526
```

즉 relevant history turn의 약 절반 정도만 선택하고 있다.

예를 들어 gold가:

```text
Gold = [1, 3]
```

인데 모델이:

```text
Pred = [3]
```

이라고 하면 질문 자체가 이해될 수도 있지만 평가상 recall은 50%다.

더 심한 경우:

```text
Gold = [1, 3, 4]
Pred = [3]
```

처럼 downstream answer에 필요한 정보를 실제로 놓칠 가능성도 있다.

따라서 지금 결과만 보면 **history_select가 context efficiency는 매우 좋지만 aggressive pruning 성향이 있다.**

---

## 7. Exact Selection이 낮은 이유

Exact match:

```text
always_all      8%
binary_gate     8%
history_select 12%
```

`history_select`가 가장 높기는 하지만 절대 수치는 낮다.

이 결과는 두 가지 가능성이 있다.

첫째, 실제로 Luna가 필요한 turn을 정확히 모두 찾지 못하고 있을 수 있다.

둘째, 현재 gold turn 생성 방식 자체가 strict할 가능성도 있다.

현재 CANARD rewrite 기반으로:

```text
rewrite에는 있고 current question에는 없는 단어
↓
그 단어가 등장하는 history turn
↓
gold relevant turn
```

방식으로 gold를 근사했다면 실제 의미상 필요한 turn과 완전히 일치하지 않을 수 있다.

예를 들어:

```text
H1: Obama became president in 2009.
H2: He was born in Hawaii.
Q : Where was he born?
```

실제로 필요한 history는 H1 또는 H2의 entity resolution 정도일 수 있는데 lexical matching은 이를 다르게 label할 수 있다.

따라서 `exact=0.12`만 보고 selector가 88% 실패한다고 해석하면 안 된다.

---

## 8. Latency

```text
binary_gate     1.222s
history_select  1.327s
```

차이는:

```text
+0.105 sec
```

약 8.6% 정도다.

selection이라는 더 어려운 task를 수행하면서도 latency 차이가 100ms 수준이므로, 이번 실험에서는 selection의 추가 latency cost가 크지 않았다.

특히 downstream 모델에 전달되는 context가 크게 줄어들기 때문에 전체 pipeline latency 기준에서는 오히려 `history_select`가 유리해질 가능성도 있다.

---

## 9. 세 방법의 trade-off

| 방법 | 장점 | 단점 |
|---|---|---|
| `always_all` | 필요한 history를 절대 놓치지 않음 | noise/context cost가 큼 |
| `binary_gate` | 구현이 단순하고 history 사용 여부를 어느 정도 판단 | `1`이면 전체 history를 넣기 때문에 context reduction이 작음 |
| `history_select` | Binary 정확도 높음, context를 매우 크게 줄임 | 필요한 history turn을 놓칠 가능성이 있음 |

---

## 10. 이번 결과에서 가장 중요한 발견

### Binary routing만 하는 것보다 direct selection이 더 잘 됨

예상과 다르게 가장 단순한

```text
0 / 1
```

classification이 최고의 router가 아니었다.

결과는:

```text
Binary Gate
Accuracy = 73%

History Selection
Accuracy = 91%
```

이다.

이는 모델이 단순히 추상적으로

> 이전 대화가 필요한가?

를 판단하는 것보다,

> 이전 대화 중 실제로 어떤 문장이 현재 질문과 연결되는가?

라는 구체적인 evidence-selection 문제를 푸는 것이 더 쉬웠을 가능성을 보여준다.

---

## 11. 현재 결과 기준 추천 구조

이번 실험만 기준으로 보면 다음 구조가 가장 유망하다.

```text
Current Query
     │
     ▼
┌──────────────────┐
│ History Selector │
│ 0 or 1,3,...     │
└────────┬─────────┘
         │
      ┌──┴────────────

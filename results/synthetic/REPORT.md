# Distillation run — SYNTHETIC demo data

Source: synthetic cards and a hand-written heuristic teacher (`veto_distill.synthetic`); no market data, no LLM. Numbers here only show the pipeline works.

Cards: 1200 teacher-labelled; students train on the earliest 60% of each group (720 cards, up to 2022-02-22) and are scored on the rest (480 cards). The teacher said TAKE on 31% of training cards.

## Summary

- **Distillation:** the best student (`pwin_tree`) agrees with the teacher on 75% of test cards (Cohen's κ 0.36); always giving the teacher's most common answer agrees on 72%.
- **Speed:** teacher median 1,260 ms per card (one API call); `pwin_tree` 0.48 ms per card including training.
- **Usefulness:** taking every test signal earns -0.075 R per trade; the teacher's TAKE set -0.081 R; `pwin_tree` -0.102 R. The teacher does not beat taking everything, so even a perfect copy inherits no edge.

## All decision makers (test cards)

| model | what it is | agreement | κ | TAKE | win rate | R net / trade | total R | ms / card |
|---|---|---|---|---|---|---|---|---|
| `take_all` | take every signal (no veto) | 27.9% | 0.00 | 480 (100%) | 29.6% | -0.075 ± 0.061 | -35.8 | – |
| `teacher` | the LLM's decisions | 100.0% | 1.00 | 134 (28%) | 29.9% | -0.081 ± 0.117 | -10.9 | 1,260.00 |
| `majority` | student: always the teacher's most common answer | 72.1% | 0.00 | 0 (0%) | – | – ± – | +0.0 | 0.00 |
| `logistic` | student: logistic regression on the card's numbers | 50.0% | -0.02 | 230 (48%) | 31.3% | -0.042 ± 0.090 | -9.7 | 0.03 |
| `tree` | student: gradient-boosted tree (depth 3) on the card's numbers | 71.2% | 0.30 | 140 (29%) | 29.3% | -0.098 ± 0.113 | -13.7 | 3.85 |
| `text` | student: char n-gram TF-IDF + logistic regression on the card text | 57.5% | 0.00 | 160 (33%) | 30.0% | -0.035 ± 0.104 | -5.7 | 1.83 |
| `pwin_tree` | student: tree regressing the teacher's p_win, then the teacher's rule | 75.2% | 0.36 | 117 (24%) | 29.9% | -0.102 ± 0.124 | -11.9 | 0.48 |
| `outcome_tree` | not a student: the same tree trained on real outcomes | 54.6% | -0.01 | 186 (39%) | 28.5% | -0.130 ± 0.098 | -24.3 | 0.31 |

`pwin_tree` reproduces the teacher's p_win with a mean absolute error of 0.036 and rank correlation 0.36.

Teacher ms / card is the API latency the teacher run logged; student ms / card is fit plus predict time on this machine divided by test cards, so it is an upper bound on inference time. R net ± is one standard error of the mean.

## Per group (test cards)

| group | cards | take all: R net / trade | teacher: TAKE | teacher: R net / trade |
|---|---|---|---|---|
| A | 240 | -0.099 | 68 | -0.035 |
| B | 240 | -0.051 | 66 | -0.130 |

## Is the teacher calibrated?

| teacher p_win (bin mean) | realised win rate | cards |
|---|---|---|
| 0.277 | 35.0% | 200 |
| 0.301 | 31.5% | 200 |
| 0.350 | 32.5% | 200 |
| 0.350 | 29.0% | 200 |
| 0.390 | 34.0% | 200 |
| 0.422 | 33.0% | 200 |

## Figures

![agreement.png](figures/agreement.png)
![r_net.png](figures/r_net.png)
![cumulative_r.png](figures/cumulative_r.png)
![calibration.png](figures/calibration.png)
![pwin_scatter.png](figures/pwin_scatter.png)

## Example cards and teacher answers

```
Signal: LONG, 15m bars. StochRSI K 7.5 -> 18.7 (turn out of oversold).
Volatility: ATR(14) = 0.48% of price; rank 36/100 over the last 30 days.
Cost: round-trip fee = 10% of the 1-ATR risk.
Trade plan: stop 1 ATR, take-profit 2 ATR, max 20 bars.
Trend: price vs EMA50 +0.52 ATR, vs EMA200 -1.51 ATR; EMA50 above EMA200.
Location: price vs daily VWAP -0.14 ATR; position in last 20 bars' range 69/100.
Last 4 bar moves (ATR, oldest first): +0.72, -0.43, -0.21, +0.69.
RSI(14) 22; volume 1.9x its 20-bar average.
Time: 05:00 UTC (Asia session).
Answer: TAKE or DONT_TAKE, win probability 0-1, reason in at most 40 words.
```
Teacher: **TAKE**, p_win 0.45. _trend supports the direction; RSI stretched; volume confirms; fee heavy._  
Outcome (never shown to any model): loss, R net -1.10.

```
Signal: LONG, 15m bars. StochRSI K 0.0 -> 10.8 (turn out of oversold).
Volatility: ATR(14) = 0.34% of price; rank 23/100 over the last 30 days.
Cost: round-trip fee = 10% of the 1-ATR risk.
Trade plan: stop 1 ATR, take-profit 2 ATR, max 20 bars.
Trend: price vs EMA50 +3.47 ATR, vs EMA200 +7.24 ATR; EMA50 below EMA200.
Location: price vs daily VWAP -1.29 ATR; position in last 20 bars' range 57/100.
Last 4 bar moves (ATR, oldest first): -0.50, -0.56, +0.32, -0.41.
RSI(14) 29; volume 0.5x its 20-bar average.
Time: 17:00 UTC (US session).
Answer: TAKE or DONT_TAKE, win probability 0-1, reason in at most 40 words.
```
Teacher: **TAKE**, p_win 0.40. _trend supports the direction; RSI stretched; fee heavy._  
Outcome (never shown to any model): win, R net +1.90.

```
Signal: SHORT, 15m bars. StochRSI K 82.5 -> 76.3 (turn out of overbought).
Volatility: ATR(14) = 0.28% of price; rank 42/100 over the last 30 days.
Cost: round-trip fee = 10% of the 1-ATR risk.
Trade plan: stop 1 ATR, take-profit 2 ATR, max 20 bars.
Trend: price vs EMA50 +1.47 ATR, vs EMA200 +0.34 ATR; EMA50 above EMA200.
Location: price vs daily VWAP +0.15 ATR; position in last 20 bars' range 6/100.
Last 4 bar moves (ATR, oldest first): +0.63, -1.11, -0.49, -0.45.
RSI(14) 56; volume 1.1x its 20-bar average.
Time: 07:00 UTC (Asia session).
Answer: TAKE or DONT_TAKE, win probability 0-1, reason in at most 40 words.
```
Teacher: **DONT_TAKE**, p_win 0.25. _against the EMA50 trend; fee heavy._  
Outcome (never shown to any model): win, R net +1.90.


# Белгілердің дәлдігі (precision)

`python -m safenet.signals` арқылы есептелген. Белгілер сөздігі тек **train** бойынша бапталды. **val** жинағы жүйелі қателерді табу үшін қаралды. **test** пен **gold** бұл кезеңде қолданылған жоқ.

Талап: әр core-белгінің өз трегіне қатысты precision-ы train жинағында ≥ 0.7.

## train (1441 жазба)

| белгі | трек | core | іске қосылды | precision (трек) | precision (gold_signals) | recall (gold_signals) |
|---|---|:-:|---:|---:|---:|---:|
| C_PAY_TRANSFER | common |  | 338 | 0.74 | 1.00 | 0.98 |
| C_PERSONAL_CARD | common |  | 139 | 0.96 | 0.96 | 1.00 |
| C_URGENCY | common |  | 232 | 0.95 | 0.89 | 0.96 |
| C_REFERRAL | common |  | 83 | 0.88 | 1.00 | 0.68 |
| C_SHORTLINK | common |  | 110 | 1.00 | 1.00 | 1.00 |
| C_DM_REDIRECT | common |  | 347 | 0.92 | 0.89 | 0.97 |
| L_PAID_ENTRY | lottery | ✔ | 250 | 0.92 | 0.91 | 0.97 |
| L_RANDOM_DRAW | lottery | ✔ | 229 | 0.89 | 1.00 | 0.95 |
| L_ORGANIZER_PROFIT | lottery |  | 46 | 1.00 | 1.00 | 1.00 |
| L_SERIES | lottery |  | 87 | 1.00 | 1.00 | 0.98 |
| K_GAMBLING_TERMS | casino | ✔ | 236 | 0.82 | 0.84 | 1.00 |
| K_WIN_BRAG | casino |  | 216 | 0.88 | 0.82 | 0.98 |
| K_PROMO_BONUS | casino |  | 147 | 0.97 | 1.00 | 0.99 |
| K_REF_TRACKING | casino |  | 31 | 1.00 | 1.00 | 1.00 |
| K_MIRROR | casino |  | 98 | 1.00 | 1.00 | 1.00 |
| K_STREAM_CTA | casino |  | 87 | 0.98 | 0.39 | 1.00 |
| P_UNREAL_RETURN | pyramid | ✔ | 262 | 0.88 | 0.95 | 0.99 |
| P_GUARANTEE | pyramid |  | 79 | 1.00 | 1.00 | 1.00 |
| P_REFERRAL_TIERS | pyramid |  | 102 | 1.00 | 1.00 | 0.96 |
| P_PAYOUT_PROOF | pyramid |  | 78 | 0.87 | 0.64 | 0.85 |
| P_ORG_STRUCTURE | pyramid |  | 111 | 1.00 | 0.76 | 0.98 |
| N_FREE_ENTRY | negative |  | 53 | 1.00 | 0.89 | 0.90 |
| N_WARNING_CONTEXT | negative |  | 91 | 1.00 | 1.00 | 1.00 |
| N_COMMERCE | negative |  | 109 | 1.00 | 0.91 | 0.92 |
| N_BANK_RATE | negative |  | 27 | 1.00 | 1.00 | 1.00 |
| N_SKILL_EVENT | negative |  | 44 | 1.00 | 0.93 | 1.00 |
| N_OFFICIAL_LOTTERY | negative |  | 13 | 1.00 | 1.00 | 1.00 |

## val (477 жазба)

| белгі | трек | core | іске қосылды | precision (трек) | precision (gold_signals) | recall (gold_signals) |
|---|---|:-:|---:|---:|---:|---:|
| C_PAY_TRANSFER | common |  | 96 | 0.64 | 0.92 | 1.00 |
| C_PERSONAL_CARD | common |  | 15 | 1.00 | 1.00 | 1.00 |
| C_URGENCY | common |  | 32 | 0.94 | 1.00 | 0.78 |
| C_REFERRAL | common |  | 8 | 1.00 | 1.00 | 0.22 |
| C_SHORTLINK | common |  | 26 | 1.00 | 1.00 | 1.00 |
| C_DM_REDIRECT | common |  | 145 | 0.91 | 0.81 | 1.00 |
| L_PAID_ENTRY | lottery | ✔ | 73 | 1.00 | 0.99 | 0.92 |
| L_RANDOM_DRAW | lottery | ✔ | 73 | 0.88 | 1.00 | 1.00 |
| L_ORGANIZER_PROFIT | lottery |  | 41 | 1.00 | 1.00 | 1.00 |
| L_SERIES | lottery |  | 12 | 1.00 | 1.00 | 0.92 |
| K_GAMBLING_TERMS | casino | ✔ | 75 | 0.88 | 0.79 | 1.00 |
| K_WIN_BRAG | casino |  | 51 | 0.72 | 0.72 | 0.93 |
| K_PROMO_BONUS | casino |  | 45 | 1.00 | 1.00 | 1.00 |
| K_REF_TRACKING | casino |  | 15 | 1.00 | 1.00 | 1.00 |
| K_MIRROR | casino |  | 14 | 1.00 | 1.00 | 1.00 |
| K_STREAM_CTA | casino |  | 9 | 1.00 | 1.00 | 0.90 |
| P_UNREAL_RETURN | pyramid | ✔ | 70 | 1.00 | 1.00 | 1.00 |
| P_GUARANTEE | pyramid |  | 40 | 0.78 | 1.00 | 1.00 |
| P_REFERRAL_TIERS | pyramid |  | 28 | 1.00 | 1.00 | 1.00 |
| P_PAYOUT_PROOF | pyramid |  | 35 | 0.94 | 0.94 | 1.00 |
| P_ORG_STRUCTURE | pyramid |  | 34 | 1.00 | 1.00 | 1.00 |
| N_FREE_ENTRY | negative |  | 23 | 1.00 | 1.00 | 1.00 |
| N_WARNING_CONTEXT | negative |  | 29 | 1.00 | 1.00 | 1.00 |
| N_COMMERCE | negative |  | 42 | 1.00 | 0.88 | 0.90 |
| N_BANK_RATE | negative |  | 9 | 1.00 | 1.00 | 1.00 |
| N_SKILL_EVENT | negative |  | 14 | 1.00 | 1.00 | 1.00 |
| N_OFFICIAL_LOTTERY | negative |  | 0 | — | — | — |

## Core-белгілер (train)

- L_PAID_ENTRY: precision = 0.92 ✅
- L_RANDOM_DRAW: precision = 0.89 ✅
- K_GAMBLING_TERMS: precision = 0.82 ✅
- P_UNREAL_RETURN: precision = 0.88 ✅

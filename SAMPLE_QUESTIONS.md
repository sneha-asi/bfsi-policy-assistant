# Sample Questions & Expected Answers

Demo script for the **BFSI Policy Assistant** (live: https://bfsi-vercel.vercel.app).

The corpus is **synthetic** policy & T&C documents for a fictional insurer/bank, *Aegis Financial*.
Every answer is grounded — the assistant cites the exact document and page, and **abstains**
(says it cannot find the answer) for anything outside these documents.

---

## Health Insurance — *Aegis Health Shield*

| Question | Expected answer |
|---|---|
| What is the waiting period for pre-existing diseases? | **36 months** of continuous coverage. (Specified illnesses like cataract, hernia, joint replacement: **24 months**; initial waiting period **30 days** except accidents.) |
| Is there a room rent limit? | Room rent capped at **1% of sum insured per day** for normal rooms, **2% for ICU**. Cataract surgery limited to **Rs. 40,000 per eye**. |
| What sum insured options are available? | **Rs. 3,00,000, Rs. 5,00,000, and Rs. 10,00,000** per policy year. |
| Is maternity covered? | Yes — after a **9-month waiting period**, up to **Rs. 50,000** for normal delivery. |
| What counts as hospitalization? | Minimum **24 consecutive hours** of in-patient care (day-care procedures in Annexure A are exempt). |

## Term Life — *Aegis SecureLife*

| Question | Expected answer |
|---|---|
| What is the suicide clause? | Death by suicide within **12 months** of commencement/revival → nominee gets **80% of premiums paid**, contract terminates. |
| What is the grace period for premiums? | **30 days** (15 days for monthly mode). After that, the policy lapses. |
| What riders are available? | **Accidental Death Benefit, Critical Illness (36 conditions), Waiver of Premium** on permanent disability. |
| What sum assured range is offered? | **Rs. 25,00,000 to Rs. 2,00,00,000**; entry age 18–65; terms of 10/20/30 years. |

## Motor — *Aegis Motor Secure*

| Question | Expected answer |
|---|---|
| How is No Claim Bonus calculated? | **20%** after year 1, then 25%, 35%, 45%, max **50%** after five claim-free years. **Forfeited** if any claim is made. |
| What is the depreciation on parts? | **50%** on rubber, plastic, and battery parts for partial loss. |
| What deductible applies? | **Rs. 1,000** per claim (≤1500cc), **Rs. 2,000** (>1500cc). |
| When is a claim not payable? | Driving without a valid licence, or under the influence of alcohol/drugs. |

## Savings Account — *Terms & Schedule of Charges*

| Question | Expected answer |
|---|---|
| What is the daily ATM withdrawal limit? | **Rs. 50,000**. (Daily UPI limit **Rs. 1,00,000**.) |
| What is the minimum balance requirement? | AMB **Rs. 10,000** (metro/urban), **Rs. 5,000** (semi-urban/rural). Non-maintenance: up to **Rs. 600/month**. |
| What interest does a savings account earn? | **3.0% p.a.** up to Rs. 1 lakh, **3.5% p.a.** above, credited quarterly. |

## Credit Card — *Cardholder Agreement*

| Question | Expected answer |
|---|---|
| What is the interest-free credit period? | Up to **50 days** if the total due is paid by the due date. |
| What are the finance charges? | **3.5% per month (42% per annum)** on unpaid balances and cash advances. |
| What is the annual fee? | **Rs. 999**, waived on annual spends of **Rs. 2,00,000+**. |
| What is the Minimum Amount Due? | **5% of total outstanding or Rs. 200**, whichever is higher, plus overlimit/unpaid amounts. |

## Fixed Deposit — *Terms & Conditions*

| Question | Expected answer |
|---|---|
| What is the premature withdrawal penalty? | **1.0%** on the applicable rate for the period held. (5-year tax-saver type: **5-year lock-in**, no premature withdrawal.) |
| What FD interest rates apply? | **4.0%–7.1% p.a.** by tenure; **+0.50%** for senior citizens. |
| When is TDS deducted? | At **10%** if interest exceeds **Rs. 40,000/year** (Rs. 50,000 for seniors), unless Form 15G/15H is submitted. |

## Home Loan — *Policy*

| Question | Expected answer |
|---|---|
| What is the maximum loan-to-value ratio? | **90%** up to Rs. 30 lakh, **80%** Rs. 30–75 lakh, **75%** above Rs. 75 lakh. |
| Are there prepayment charges? | **No** prepayment/foreclosure charges on floating-rate loans taken by individuals. |
| What is the eligibility age? | Salaried **21–60**, self-employed **25–65**; min net monthly income **Rs. 30,000**. |

## Fraud & Dispute (SCANNED PDF — exercises the OCR path)

| Question | Expected answer |
|---|---|
| How long do I have to report a fraudulent transaction? | Within **3 working days** → **zero customer liability**. |
| How fast is a dispute resolved? | Shadow credit within **10 working days** where the customer is not at fault. |
| How long do chargebacks take? | Up to **45 days** through the card network. |
| Will the bank ask for my OTP/PIN/CVV? | **Never.** Sharing them may shift liability to the customer. |

## Claims SLA · KYC/AML · Grievance · Privacy

| Question | Expected answer |
|---|---|
| How long does a claim decision take? | Acknowledged in **24 hrs**, surveyor in **48 hrs**, decision in **15 working days**, payment **7 days** after. |
| How often is KYC reviewed? | High-risk/PEP **every 2 years**, medium **8 years**, low **10 years**. |
| What is the grievance escalation path? | L1 Customer Care (48h) → L2 Grievance Officer (10 working days) → L3 **Ombudsman** (after 30 days, free). |
| How long is my data retained? | Minimum **5 years** after the relationship ends, then securely deleted/anonymised. |

---

## Out-of-scope (the assistant should ABSTAIN)

These prove the grounding/anti-hallucination gate. Expected response: a polite refusal that
the answer is not in the loaded policy documents — **no guessing**.

- *Who won the 2018 World Cup?*
- *What is the capital of France?*
- *Can you write me a poem about insurance?*
- *What is Aegis Financial's stock price?* (not in the docs)

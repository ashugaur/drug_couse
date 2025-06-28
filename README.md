# Drug couse

Method(s) to analyze patient drug couse.


## Associated rule mining

`Question`: Likelihood of a patient taking drug A also taking drug B?


### 1. Association Rule Mining (ARM)

ARM is an `Apriori Algorithm` similar to market basket analysis where you can identify associations between different drugs. You can look for rules like "if a patient takes Drug A, they are likely to take Drug B".

1. `Support`

    How frequently a particular combination of drugs occurs.

    Support(A,B) = Count(A and B) / Total patients

    Support = 0.3 means 30% of patients take both drugs


2. `Confidence`

    Probability (likelihood) that Drug B is taken given Drug A is taken.

    Confidence(A→B) = Count(A and B) / Count(A)

    Confidence = 0.8 means 80% of patients who take Drug A also take Drug B


3. `Lift`

    Measures the deviation from independence. How much more likely Drug B is when Drug A is present vs. baseline (random chance).

    Lift(A→B) = Confidence(A→B) / Support(B)

    Lift = 1: No association (random)<br>
    Lift > 1: Positive association (Drug A increases likelihood of Drug B)<br>
    Lift < 1: Negative association (Drug A decreases likelihood of Drug B)<br>
    Lift = 2.0: Patients taking Drug A are 2x more likely to take Drug B than average


4. `Conviction`

    How much more often the rule would be incorrect if Drug A and Drug B were independent.

    Higher values: Stronger rules


**What to Look For**

- High Confidence Rules (>0.7): Strong predictive relationships
- High Lift Rules (>2.0): Interesting associations beyond random chance
- Reasonable Support (>0.1): Patterns that affect a meaningful number of patients


import pandas as pd

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["PromotionGap"] = df["YearsSinceLastPromotion"]
    df["SatisfactionWorkloadRatio"] = df["JobSatisfaction"] / (df["WorkLifeBalance"] + 1)
    df["CareerStagnation"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    df["IncomeToAgeRatio"] = df["MonthlyIncome"] / (df["Age"] + 1)
    df["ExperienceToPromotionRatio"] = df["TotalWorkingYears"] / (df["YearsSinceLastPromotion"] + 1)

    return df
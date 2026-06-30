
import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import os

def segment_players(season="20/21"):
    try:
        if not os.path.exists("no_nans_data.xlsx"):
            raise FileNotFoundError("The data file 'no_nans_data.xlsx' was not found.")
        df = pd.read_excel("no_nans_data.xlsx")
    except FileNotFoundError as e:
        print(e)
        return pd.DataFrame()

    df = df.copy()

    if season == "20/21":
        seasons_to_drop = ["17/18", "18/19", "19/20"]
    elif season == "19/20":
        seasons_to_drop = ["17/18", "18/19", "20/21"]
    else:
        raise ValueError("Invalid season specified. Please choose '19/20' or '20/21'.")

    for s in seasons_to_drop:
        keyword = f"({s})"
        columns_to_drop = [col for col in df.columns if keyword in col]
        df.drop(columns=columns_to_drop, inplace=True)

    # ... (rest of the function remains the same)
    # DEFENDER SEGMENTATION
    df_defender = df[df["Position"] == "Defender"].copy()
    defender_columns = [col for col in df.columns if col in ["Position","Age","Player","Nation",f"MP ({season})",
f"Min ({season})", f"Goals/Shots ({season})",f"Passes Leading to Shot Attempt ({season})",f"Defensive Actions Leading to Shot Attempt ({season})", 
f"Touches in Defensive Penalty Box ({season})", f"Touches in Defensive 3rd ({season})",f"Touches in Midfield 3rd ({season})",f"Touches in Open-play ({season})",
f"Total Carries ({season})" ,f"Total Distance Carried the Ball ({season})",f"% of Times Successfully Received Pass ({season})" ,f"Pass Completion % (All pass-types) ({season})",
f"Total Tackles Won ({season})",f"Total Defensive Blocks ({season})",f"Total Shots Blocked ({season})",f"Goal Saving Blocks ({season})", 
f"Times blocked a Pass ({season})", f"Aerial Duel Won ({season})",f"Aerial Duel Lost ({season})",f"Total Loose Balls Recovered ({season})"]]
    df_defender = df_defender[defender_columns]

    df_defender["Aerial_Duel_Total"] = df_defender[f"Aerial Duel Won ({season})"] - df_defender[f"Aerial Duel Lost ({season})"]

    df_defender["Age_Score"] = pd.qcut(df_defender["Age"], 5, labels= [5,4,3,2,1])
    df_defender["MP_Score"] = pd.qcut(df_defender[f"MP ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["Min_Score"] = pd.qcut(df_defender[f"Min ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["Passes_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_defender[f"Passes Leading to Shot Attempt ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["Defensive_Actions_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_defender[f"Defensive Actions Leading to Shot Attempt ({season})"].rank(method="first"), 5, labels= [1,2,3,4,5])
    df_defender["Touches_in_Defensive_Penalty_Box_Score"] = pd.qcut(df_defender[f"Touches in Defensive Penalty Box ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["Touches_in_Defensive_3rd_Score"] = pd.qcut(df_defender[f"Touches in Defensive 3rd ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["Total_Disctance_Score"] = pd.qcut(df_defender[f"Total Distance Carried the Ball ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["Pass_Completion_Score"] = pd.qcut(df_defender[f"Pass Completion % (All pass-types) ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["%_of_Times_Successfully_Received_Pass_Score"] = pd.qcut(df_defender[f"% of Times Successfully Received Pass ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["total_Loose_Balls_Recovered_Score"] = pd.qcut(df_defender[f"Total Loose Balls Recovered ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["Total_Tackles_Won_Score"] = pd.qcut(df_defender[f"Total Tackles Won ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["Total_Defensive_Blocks_Score"] = pd.qcut(df_defender[f"Total Defensive Blocks ({season})"], 5, labels= [1,2,3,4,5])
    df_defender["Aerial_Duel_Total_Score"] = pd.qcut(df_defender["Aerial_Duel_Total"], 5, labels= [1,2,3,4,5])

    df_defender["Total_Score"] = (df_defender["MP_Score"].astype(int) +  df_defender["Min_Score"].astype(int) + 
                                  df_defender["Passes_Leading_to_Shot_Attempt_Score"].astype(int) + df_defender["Defensive_Actions_Leading_to_Shot_Attempt_Score"].astype(int) + 
                                  df_defender["Touches_in_Defensive_Penalty_Box_Score"].astype(int) + df_defender["Touches_in_Defensive_3rd_Score"].astype(int) + 
                                  df_defender["%_of_Times_Successfully_Received_Pass_Score"].astype(int) + df_defender["Total_Tackles_Won_Score"].astype(int) + 
                                  df_defender["Total_Defensive_Blocks_Score"].astype(int) + df_defender["Aerial_Duel_Total_Score"].astype(int) +
                                  df_defender["Total_Disctance_Score"].astype(int) + df_defender["Pass_Completion_Score"].astype(int) + df_defender["total_Loose_Balls_Recovered_Score"].astype(int))

    if season == "20/21":
        df_defender['Performance'] = pd.cut(x=df_defender['Total_Score'], bins=[13, 24, 39, 49, 58, 63],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])
    else:
        df_defender['Performance'] = pd.cut(x=df_defender['Total_Score'], bins=[13, 24, 39, 49, 57, 63],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])

    df_defender['Age_Cat'] = pd.cut(x=df_defender['Age'], bins=[17, 22, 27, 32, 37],labels=["Young", "Experienced", "Mature", "End_Of_Career" ])
    df_defender[f"Segment{season.replace('/', '_')}"] = df_defender["Age_Cat"].astype(str) + "_" + df_defender["Performance"].astype(str)

    performance_score_col = f"Performance_Score_{season.replace('/', '_')}"
    df_defender.loc[(df_defender["Performance"] == "Under_expected"), performance_score_col] = "1"
    df_defender.loc[(df_defender["Performance"] == "Open_to_development"), performance_score_col] = "2"
    df_defender.loc[(df_defender["Performance"] == "Player_with_high_potential"), performance_score_col] = "3"
    df_defender.loc[(df_defender["Performance"] == "High_performance"), performance_score_col] = "4"
    df_defender.loc[(df_defender["Performance"] == "Flawless"), performance_score_col] = "5"

    final_defender = df_defender[["Player", f"Segment{season.replace('/', '_')}", performance_score_col]].copy()

    # MIDFIELD SEGMENTATION
    df_midfield = df[df["Position"] == "midfield"].copy()
    midfield_columns = [col for col in df.columns if col in ["Player","Age","Position",f"Total Distance of Completed Progressive Passes (All Pass-types) ({season})",
f"Ast ({season})", f"Gls ({season})", f"Non-Penalty Goals ({season})",f"Min ({season})",f"Shots on Target% ({season})", f"Goals/Shots on Target ({season})",f"Passes Leading to Shot Attempt ({season})",
f"Dribbles Leading to Shot Attempt ({season})",f"Defensive Actions Leading to Shot Attempt ({season})" ,f"Passes Leading to Goals ({season})", 
f"Touches in Midfield 3rd ({season})", f"Touches in Attacking 3rd ({season})",f"Touches in Defensive 3rd ({season})", f"Touches in Open-play ({season})",
f"Total Distance Carried the Ball ({season})",f"Total Distance Carried the Ball in Forward Direction","Number of Times Player was Pass Target" ,
f"% of Times Successfully Received Pass ({season})"  ,f"Pass Completion % (All pass-types) ({season})",f"Completed passes that enter Final 3rd ({season})",
f"Total Tackles Won ({season})",f"Total Players Tackled + Total Interceptions ({season})",f"MP ({season})","Club","Nation","League"]]
    df_midfield = df_midfield[midfield_columns]

    df_midfield["Age_Score"] = pd.qcut(df_midfield["Age"], 5, labels= [5,4,3,2,1])
    df_midfield["MP_Score"] = pd.qcut(df_midfield[f"MP ({season})"], 5, labels= [1,2,3,4,5])
    df_midfield["Min_Score"] = pd.qcut(df_midfield[f"Min ({season})"], 5, labels= [1,2,3,4,5])
    df_midfield["Ast_Score"] = pd.qcut(df_midfield[f"Ast ({season})"].rank(method="first"), 5, labels= [1,2,3,4,5])
    df_midfield["Gls_Score"] = pd.qcut(df_midfield[f"Gls ({season})"].rank(method="first"), 5, labels= [1,2,3,4,5])
    df_midfield["Non-Penalty_goals_Score"] = pd.qcut(df_midfield[f"Non-Penalty Goals ({season})"].rank(method="first"), 5, labels= [1,2,3,4,5])
    df_midfield["Goals/Shots_on_Target_Score"] = pd.qcut(df_midfield[f"Goals/Shots on Target ({season})"].rank(method = "first"), 5, labels= [1,2,3,4,5])
    df_midfield["PassesLeadingtoShotAttempt_Score"] = pd.qcut(df_midfield[f"Passes Leading to Shot Attempt ({season})"], 5, labels= [1,2,3,4,5])
    df_midfield["Defensive_Actions_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_midfield[f"Defensive Actions Leading to Shot Attempt ({season})"].rank(method = "first"), 5, labels= [1,2,3,4,5])
    df_midfield["TouchesinDefensive3rd_score"] = pd.qcut(df_midfield[f"Touches in Defensive 3rd ({season})"], 5, labels= [1,2,3,4,5])
    df_midfield["Touches in Midfield 3rd_Score"] = pd.qcut(df_midfield[f"Touches in Midfield 3rd ({season})"], 5, labels= [1,2,3,4,5])
    df_midfield["Touches in Attacking 3rd_Score"] = pd.qcut(df_midfield[f"Touches in Attacking 3rd ({season})"], 5, labels= [1,2,3,4,5])
    df_midfield["Touches in Open-play_Score"] = pd.qcut(df_midfield[f"Touches in Open-play ({season})"], 5, labels= [1,2,3,4,5])
    df_midfield["Number of Times Player was Pass Target_Score"] = pd.qcut(df_midfield["Number of Times Player was Pass Target"], 5, labels= [1,2,3,4,5])
    df_midfield["Pass Completion % (All pass-types)_Score"] = pd.qcut(df_midfield[f"Pass Completion % (All pass-types) ({season})"], 5, labels= [1,2,3,4,5])
    df_midfield["Completed passes that enter Final 3rd_Score"] = pd.qcut(df_midfield[f"Completed passes that enter Final 3rd ({season})"], 5, labels= [1,2,3,4,5])
    df_midfield["Total Tackles Won_Score"] = pd.qcut(df_midfield[f"Total Tackles Won ({season})"], 5, labels= [1,2,3,4,5])

    df_midfield["Total_Score"] = (df_midfield["MP_Score"].astype(int) +  df_midfield["Min_Score"].astype(int) + 
                                  df_midfield["Ast_Score"].astype(int) + df_midfield["Gls_Score"].astype(int) + 
                                  df_midfield["Non-Penalty_goals_Score"].astype(int) + df_midfield["Goals/Shots_on_Target_Score"].astype(int) + 
                                  df_midfield["PassesLeadingtoShotAttempt_Score"].astype(int) + df_midfield["Defensive_Actions_Leading_to_Shot_Attempt_Score"].astype(int)+ 
                                  df_midfield["TouchesinDefensive3rd_score"].astype(int) + df_midfield["Touches in Midfield 3rd_Score"].astype(int) + 
                                  df_midfield["Touches in Attacking 3rd_Score"].astype(int) + df_midfield["Touches in Open-play_Score"].astype(int) + df_midfield["Number of Times Player was Pass Target_Score"].astype(int)+ 
                                  df_midfield["Pass Completion % (All pass-types)_Score"].astype(int) + df_midfield["Completed passes that enter Final 3rd_Score"].astype(int)+ 
                                  df_midfield["Total Tackles Won_Score"].astype(int))

    if season == "20/21":
        df_midfield['Performance'] = pd.cut(x=df_midfield['Total_Score'], bins=[16,32,45,56,68,77],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])
    else:
        df_midfield['Performance'] = pd.cut(x=df_midfield['Total_Score'], bins=[15,32,45,56,67,78],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])

    df_midfield['Age_Cat'] = pd.cut(x=df_midfield['Age'], bins=[17, 22, 27, 32, 37],labels=["Young", "Experienced", "Mature", "End_Of_Career" ])
    df_midfield[f"Segment{season.replace('/', '_')}"] = df_midfield["Age_Cat"].astype(str) + "_" + df_midfield["Performance"].astype(str)

    df_midfield.loc[(df_midfield["Performance"] == "Under_expected"), performance_score_col] = "1"
    df_midfield.loc[(df_midfield["Performance"] == "Open_to_development"), performance_score_col] = "2"
    df_midfield.loc[(df_midfield["Performance"] == "Player_with_high_potential"), performance_score_col] = "3"
    df_midfield.loc[(df_midfield["Performance"] == "High_performance"), performance_score_col] = "4"
    df_midfield.loc[(df_midfield["Performance"] == "Flawless"), performance_score_col] = "5"

    final_midfield = df_midfield[["Player", f"Segment{season.replace('/', '_')}", performance_score_col]].copy()

    # ATTACK SEGMENTATION
    df_attack = df[df["Position"] == "attack"].copy()
    attack_columns = [col for col in df.columns if col in ["Age", f"MP ({season})","Player","Club","Position","Nation","League",f"Min ({season})", 
                                                       f"Gls ({season})",f"Ast ({season})",f"Non-Penalty Goals ({season})",f"Shots on Target% ({season})",
                                                       f"Goals/Shots ({season})",f"Goals Scored minus xG ({season})",f"Passes Leading to Shot Attempt ({season})",
                                                       f"Dribbles Leading to Shot Attempt ({season})",f"Goal Creating Actions ({season})",f"Passes Leading to Goals ({season})",
                                                       f"Dribbles Leading to Goals ({season})",f"Touches in Attacking 3rd ({season})",f"Touches in Attacking Penalty Box ({season})",
                                                       f"Total Successful Dribbles ({season})",f"Carries into Attacking Penalty Box ({season})",f"Total Failed Attempts at Controlling Ball ({season})", 
                                                       f"% of Times Successfully Received Pass ({season})",f"Progressive Passes Received ({season})",f"Completed passes that enter Penalty Box ({season})",
                                                       f"Aerial Duel Won ({season})",f"Aerial Duel Lost ({season})",f"Tackles in Attacking 3rd ({season})",f"Successful Pressure % ({season})"]]
    df_attack = df_attack[attack_columns]

    df_attack["Aerial_Duel_Total"] = df_attack[f"Aerial Duel Won ({season})"] - df_attack[f"Aerial Duel Lost ({season})"]

    df_attack["Age_Score"] = pd.qcut(df_attack["Age"], 5, labels= [5,4,3,2,1])
    df_attack["MP_Score"] = pd.qcut(df_attack[f"MP ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Min_Score"] = pd.qcut(df_attack[f"Min ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Ast_Score"] = pd.qcut(df_attack[f"Ast ({season})"].rank(method="first"), 5, labels= [1,2,3,4,5])
    df_attack["Gls_Score"] = pd.qcut(df_attack[f"Gls ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Non-Penalty_goals_Score"] = pd.qcut(df_attack[f"Non-Penalty Goals ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Shots_on_Target%_Score"] = pd.qcut(df_attack[f"Shots on Target% ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Goals/Shots_Score"] = pd.qcut(df_attack[f"Goals/Shots ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Goal_scored_minus_xG_Score"] = pd.qcut(df_attack[f"Goals Scored minus xG ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Passes_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_attack[f"Passes Leading to Shot Attempt ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Dribbles_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_attack[f"Dribbles Leading to Shot Attempt ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Goal_Creating_Actions_Score"] = pd.qcut(df_attack[f"Goal Creating Actions ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Passes_Leading_to_Goals_Score"] = pd.qcut(df_attack[f"Passes Leading to Goals ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Dribbles_Leading_to_Goals_Score"] = pd.qcut(df_attack[f"Dribbles Leading to Goals ({season})"].rank(method="first"), 5, labels= [1,2,3,4,5])
    df_attack["Touches_in_Attacking_3rd_Score"] = pd.qcut(df_attack[f"Touches in Attacking 3rd ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Touches_in_Attacking_Penalty_Box_Score"] = pd.qcut(df_attack[f"Touches in Attacking Penalty Box ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Carries_into_Attacking_Penalty_Box_Score"] = pd.qcut(df_attack[f"Carries into Attacking Penalty Box ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Total_Successful_Dribbles_Score"] = pd.qcut(df_attack[f"Total Successful Dribbles ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Total_Failed_Attempts_at_Controlling_Ball_Score"] = pd.qcut(df_attack[f"Total Failed Attempts at Controlling Ball ({season})"].rank(method="first"), 5, labels= [5,4,3,2,1])
    df_attack["%Progressive_Passes_Received_Score"] = pd.qcut(df_attack[f"Progressive Passes Received ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["%Completed_passes_that_enter_Penalty_Box_Score"] = pd.qcut(df_attack[f"Completed passes that enter Penalty Box ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["%Aerial_Duel_Total_Score"] = pd.qcut(df_attack["Aerial_Duel_Total"], 5, labels= [1,2,3,4,5])
    df_attack["%_of_Times_Successfully_Received_Pass_Score"] = pd.qcut(df_attack[f"% of Times Successfully Received Pass ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Tackles_in_Attacking_3rd_Score"] = pd.qcut(df_attack[f"Tackles in Attacking 3rd ({season})"], 5, labels= [1,2,3,4,5])
    df_attack["Successful_Pressure_%_Score"] = pd.qcut(df_attack[f"Successful Pressure % ({season})"], 5, labels= [1,2,3,4,5])

    df_attack["Total_Score"] = (df_attack["MP_Score"].astype(int) +  df_attack["Min_Score"].astype(int) + 
                                  df_attack["Ast_Score"].astype(int) + df_attack["Gls_Score"].astype(int) + 
                                  df_attack["Non-Penalty_goals_Score"].astype(int) + df_attack["Shots_on_Target%_Score"].astype(int) + 
                                  df_attack["Goals/Shots_Score"].astype(int) + df_attack["Goal_scored_minus_xG_Score"].astype(int)+ 
                                  df_attack["Passes_Leading_to_Shot_Attempt_Score"].astype(int) + df_attack["Dribbles_Leading_to_Goals_Score"].astype(int) + 
                                  df_attack["Goal_Creating_Actions_Score"].astype(int) + df_attack["Passes_Leading_to_Goals_Score"].astype(int) + df_attack["Dribbles_Leading_to_Goals_Score"].astype(int)+ 
                                  df_attack["Touches_in_Attacking_3rd_Score"].astype(int) + df_attack["Touches_in_Attacking_Penalty_Box_Score"].astype(int)+ 
                                  df_attack["Carries_into_Attacking_Penalty_Box_Score"].astype(int) + df_attack["Total_Successful_Dribbles_Score"].astype(int) +df_attack["Total_Failed_Attempts_at_Controlling_Ball_Score"].astype(int)+ 
                                  df_attack["%Progressive_Passes_Received_Score"].astype(int) + df_attack["%Completed_passes_that_enter_Penalty_Box_Score"].astype(int) +df_attack["%Aerial_Duel_Total_Score"].astype(int) + 
                                  df_attack["%_of_Times_Successfully_Received_Pass_Score"].astype(int)  + df_attack["Tackles_in_Attacking_3rd_Score"].astype(int) + df_attack["Successful_Pressure_%_Score"].astype(int) )

    df_attack['Performance'] = pd.cut(x=df_attack['Total_Score'], bins=[33,50,72,86,100,113],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])
    df_attack['Age_Cat'] = pd.cut(x=df_attack['Age'], bins=[15, 22, 27, 32, 40],labels=["Young", "Experienced", "Mature", "End_Of_Career" ])
    df_attack[f"Segment{season.replace('/', '_')}"] = df_attack["Age_Cat"].astype(str) + "_" + df_attack["Performance"].astype(str)

    df_attack.loc[(df_attack["Performance"] == "Under_expected"), performance_score_col] = "1"
    df_attack.loc[(df_attack["Performance"] == "Open_to_development"), performance_score_col] = "2"
    df_attack.loc[(df_attack["Performance"] == "Player_with_high_potential"), performance_score_col] = "3"
    df_attack.loc[(df_attack["Performance"] == "High_performance"), performance_score_col] = "4"
    df_attack.loc[(df_attack["Performance"] == "Flawless"), performance_score_col] = "5"

    final_attack = df_attack[["Player", f"Segment{season.replace('/', '_')}", performance_score_col]].copy()

    final_total_df = pd.concat([final_defender, final_midfield, final_attack], axis=0)

    # Sales Expectation
    segment_col = f"Segment{season.replace('/', '_')}"
    final_total_df.loc[(final_total_df[segment_col] == "Young_Under_expected"), "Sales_Expectation_Price"] = "Low"
    final_total_df.loc[(final_total_df[segment_col] == "Young_Open_to_development"), "Sales_Expectation_Price"] = "Low - Mid"
    final_total_df.loc[(final_total_df[segment_col] == "Young_Player_with_high_potential"), "Sales_Expectation_Price"] = "Mid"
    final_total_df.loc[(final_total_df[segment_col] == "Young_High_performance"), "Sales_Expectation_Price"] = "High"
    final_total_df.loc[(final_total_df[segment_col] == "Young_Flawless"), "Sales_Expectation_Price"] = "Very_High"

    final_total_df.loc[(final_total_df[segment_col] == "Experienced_Under_expected"), "Sales_Expectation_Price"] = "Low"
    final_total_df.loc[(final_total_df[segment_col] == "Experienced_Open_to_development"), "Sales_Expectation_Price"] = "Low - Mid"
    final_total_df.loc[(final_total_df[segment_col] == "Experienced_Player_with_high_potential"), "Sales_Expectation_Price"] = "Mid"
    final_total_df.loc[(final_total_df[segment_col] == "Experienced_High_performance"), "Sales_Expectation_Price"] = "Mid - High"
    final_total_df.loc[(final_total_df[segment_col] == "Experienced_Flawless"), "Sales_Expectation_Price"] = "Very_High"

    final_total_df.loc[(final_total_df[segment_col] == "Mature_Under_expected"), "Sales_Expectation_Price"] = "Very_Low"
    final_total_df.loc[(final_total_df[segment_col] == "Mature_Open_to_development"), "Sales_Expectation_Price"] = "Low"
    final_total_df.loc[(final_total_df[segment_col] == "Mature_Player_with_high_potential"), "Sales_Expectation_Price"] = "Mid"
    final_total_df.loc[(final_total_df[segment_col] == "Mature_High_performance"), "Sales_Expectation_Price"] = "Mid - High"
    final_total_df.loc[(final_total_df[segment_col] == "Mature_Flawless"), "Sales_Expectation_Price"] = "High"

    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_Under_expected"), "Sales_Expectation_Price"] = "Very_Low"
    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_Open_to_development"), "Sales_Expectation_Price"] = "Low"
    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_Player_with_high_potential"), "Sales_Expectation_Price"] = "Low"
    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_High_performance"), "Sales_Expectation_Price"] = "Mid"
    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_Flawless"), "Sales_Expectation_Price"] = "Mid - High"

    # Recommend For Action
    final_total_df.loc[(final_total_df[segment_col] == "Young_Under_expected"), "Recommend_For_Action"] = "Considering the potential of these young players, encourage them to improve their performance with extra work and training. By taking a long-term approach, support their development and show patience."
    final_total_df.loc[(final_total_df[segment_col] == "Young_Open_to_development"), "Recommend_For_Action"] = "Create special training programs to maximize the potential of these young players. Help them gain experience by regularly giving them chances in first team matches."
    final_total_df.loc[(final_total_df[segment_col] == "Young_Player_with_high_potential"), "Recommend_For_Action"] = "Young and high-potential players can be an important part of the team in the future. Focusing on opportunities for these players to develop their skills and gain experience can greatly benefit in the long run."
    final_total_df.loc[(final_total_df[segment_col] == "Young_High_performance"), "Recommend_For_Action"] = "Help these young talents improve their physical condition and technical skills so that they can maintain their high performance. Encourage them to show that they are ready to give leadership roles within the team."
    final_total_df.loc[(final_total_df[segment_col] == "Young_Flawless"), "Recommend_For_Action"] = "Help these young talents maximize their physical and technical abilities so that they can maintain their excellent performance. Encourage them to evaluate media and marketing opportunities to gain more visibility."

    final_total_df.loc[(final_total_df[segment_col] == "Experienced_Under_expected"), "Recommend_For_Action"] = "Create individual training plans for experienced players to improve their performance and increase their motivation. Based on their past accomplishments, allow them to focus more on the team leadership role."
    final_total_df.loc[(final_total_df[segment_col] == "Experienced_Open_to_development"), "Recommend_For_Action"] = "Take a long-term approach to preparing these experienced players for the future. Encourage them to build mentoring relationships with young players and share their knowledge."
    final_total_df.loc[(final_total_df[segment_col] == "Experienced_Player_with_high_potential"), "Recommend_For_Action"] = "Experienced and high-potential players can make an immediate contribution to the team. Thanks to the experience they have, these players can lead in tough matches and guide young players. At the same time, making a special effort to further develop the potential of these players can increase the team's chances of success"
    final_total_df.loc[(final_total_df[segment_col] == "Experienced_High_performance"), "Recommend_For_Action"] = "Support experienced players to maintain a high level of performance. Consider increasing their leadership as one of the key players on the team, giving them more responsibility within the team."
    final_total_df.loc[(final_total_df[segment_col] == "Experienced_Flawless"), "Recommend_For_Action"] = "Support experienced players to maintain flawless performances and strengthen their leadership roles. Strategically engage them to enable them to take on more responsibility within the team."

    final_total_df.loc[(final_total_df[segment_col] == "Mature_Under_expected"), "Recommend_For_Action"] = "Develop a special rehabilitation and training program to help mature players return to their jerseys. Set new goals to boost their motivation and rekindle their desire to improve their performance."
    final_total_df.loc[(final_total_df[segment_col] == "Mature_Open_to_development"), "Recommend_For_Action"] = "Create individual training and training plans to enable these mature players to develop further. Consider giving leadership roles within the team and encourage them to share their experiences with younger players."
    final_total_df.loc[(final_total_df[segment_col] == "Mature_Player_with_high_potential"), "Recommend_For_Action"] = "Create a strategic plan to maximize the high potential of these mature players. Ensure that they maintain the proper balance of training and rest so that they can maintain their performance."
    final_total_df.loc[(final_total_df[segment_col] == "Mature_High_performance"), "Recommend_For_Action"] = "Help mature players maintain their high level of performance and encourage them to make more impact within the team by increasing their leadership. Encourage young players to share their experiences by mentoring them."
    final_total_df.loc[(final_total_d_df[segment_col] == "Mature_Flawless"), "Recommend_For_Action"] = "Help mature players maintain flawless performances and encourage them to make more impact within the team by increasing their leadership. Encourage young players to share their experiences by mentoring them"

    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_Under_expected"), "Recommend_For_Action"] = "Provide specific support and motivation for players nearing the end of their careers to rotate their jerseys. Consider mentoring roles or assistant coaching positions to allow the team to benefit from their experience."
    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_Open_to_development"), "Recommend_For_Action"] = "Help players nearing the end of their careers prepare their final stages for the future of the team. Support players in thinking about their own post-career plans and training."
    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_Player_with_high_potential"), "Recommend_For_Action"] = "Players who are nearing the end of their careers but still have high potential can be a valuable asset to teams. Thanks to their experience, these players can give advice to young talents and increase  the morale of the team with their leadership on the field. The future contributions of these players must be carefully evaluated and aligned with the team's overall strategy."
    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_High_performance"), "Recommend_For_Action"] = "Provide physical and psychological support to help these players maintain their performance as they approach the end of their careers. Take full advantage of their experience by increasing their leadership role within the team."
    final_total_df.loc[(final_total_df[segment_col] == "End_Of_Career_Flawless"), "Recommend_For_Action"] = "Help players near the end of their careers maintain flawless performances and strengthen their leadership roles. Prepare players for mentoring or managerial roles to support their post-career plans."

    return final_total_df
if __name__ == '__main__':
    segment_players("20/21").to_excel('veriler_20_21.xlsx', index=False)
    segment_players("19/20").to_excel('veriler_19_20.xlsx', index=False)

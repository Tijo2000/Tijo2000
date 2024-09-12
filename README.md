### Hi there 👋

**Tijo2000/Tijo2000** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

https://accounts.google.com/ServiceLogin?passive=1209600&continue=https://cloud.google.com/innovators/getcertified?utm_content%3Dinvite1_marketo%26utm_source%3Dsales_contacts%26utm_medium%3Demail%26utm_campaign%3DFY24-Q2-global-PROD917-website-su-GetCertified-Q2%26mkt_tok%3DODA4LUdKVy0zMTQAAAGShQcBnucEH-H3mkPjgfU1gNYvccLdJhGDipdkpUqrK5M6u6-bNaVidrZXoOMo37calUKVR4gh9MFtuCyJY52vugfUzKS5wcHKrmbAQE0SdLFPxBXVLrQ&followup=https://cloud.google.com/innovators/getcertified?utm_content%3Dinvite1_marketo%26utm_source%3Dsales_contacts%26utm_medium%3Demail%26utm_campaign%3DFY24-Q2-global-PROD917-website-su-GetCertified-Q2%26mkt_tok%3DODA4LUdKVy0zMTQAAAGShQcBnucEH-H3mkPjgfU1gNYvccLdJhGDipdkpUqrK5M6u6-bNaVidrZXoOMo37calUKVR4gh9MFtuCyJY52vugfUzKS5wcHKrmbAQE0SdLFPxBXVLrQ


https://jftrgoogle.webex.com/jftrgoogle/j.php?MTID=md4aeb5e91664c09c5cf97fc4cf3fb4ea

Google group : https://ind01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgroups.google.com%2Fd%2Fforum%2Fget_certified-11_2024_q2&data=05%7C02%7Ctijo.thomas%40edelweisstokio.in%7Cfa4064b8c2564041e27608dc74cf0112%7C16a6cf82ea8449e5a55db65a9a2100df%7C0%7C0%7C638513679634475306%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=QqA6XGrbf94%2Bc93t3K5ZxKCcjmMxykkq4mMXMZh0T6U%3D&reserved=0

Troubleshoot mail : https://ind01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgo.cloudplatformonline.com%2FODA4LUdKVy0zMTQAAAGTALbd6jojCjXid2-mR6cu_0-IXQ2e1NZJum3NuJAKtTTDUnGsKs4zJLEQvA9cdQaQ4_KFyj8%3D&data=05%7C02%7Ctijo.thomas%40edelweisstokio.in%7Cd41df67d6347428d03cb08dc70d00ff9%7C16a6cf82ea8449e5a55db65a9a2100df%7C0%7C0%7C638509286139018590%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=MDUGO2SZqBI1etSVM%2FHPdgFaiSZmW7WFc7s3Z8rNyrA%3D&reserved=0

https://www.cloudskillsboost.google/catalog?qlcampaign=1m-ggcfs-79

https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Fwww%2Efictionpies%2Ecom&urlhash=93tD&trk=public_profile_topcard-website


https://forms.gle/LZNEDVzAAAPo8PgH7

2nd stage
day1 14TH jUNE : meet.google.com/mfs-ytxw-qhg
drive link : [https://docs.google.com/forms/d/e/1FAIpQLSf6bw3TAfkDKQbnLNDNw6PRbWsBJBO2mcAkzXiyigf8wwpbXg/viewform?resourcekey=0-SDojBSiWa4Pj5UK_TOjGqA](https://drive.google.com/corp/drive/folders/1fmI3whZ_pWqjdC76vGZU-aidoYz2fqt2?resourcekey=0-Dl5bLW43zcz-SdiCQASJbw)

Online group : https://groups.google.com/access-error?continue=https://groups.google.com/g/ace10-getcert_q2_2024


https://resolve-prod.lenovo.com/Lenovo-Field-Services-0.0.1-Resolve/ResolveAp/checkoutConfirmation/FS240929551004267/c248b6ba-5185-4566-a334-b2d4ff2fc0a5/true




Hi i am writing the inference code to predict propensity of agents being active in T+1, T+2 and T+3. Can you train three catboost models for each target with these params and assign the predictions to each agent . You can use this code which I use while experimenting for reference but dont  y_test, coz there's no actual labels to evaluate . X_train = base_data[variables]
X_test = val_data[variables]
for target in ['T+1', 'T+2', 'T+3']:
    print(f"\n--- CatBoost model for {target} ---\n")
    y_train = base_data[target]
    y_test = val_data[target]
    # CatBoost cat_params
    cat_params = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'scale_pos_weight': [1, 2, 3]  # Added to boost recall
    }
    cat = CatBoostClassifier(random_state=42, verbose=0)
    cat_cv = RandomizedSearchCV(cat, cat_params, n_iter=10, cv=3, scoring='recall', n_jobs=-1, random_state=42)
    cat_cv.fit(X_train, y_train)  # Fit the model first
    # Print best parameters
    print(f"Best parameters for {target}:\n{cat_cv.best_params_}")
    best_cat = train_evaluate_model(cat_cv.best_estimator_, X_train, y_train, X_test, y_test, "CatBoost"). Please generate the propensity for each target . I have stored the agent code values seperately before processing as val_agent_code, You can later store them val_data['T+1pred','T+2pred','T+3pred'] = output
val_data['AGENT_CODE'] = val_agent_code. My final data is final_data you can later merge these three propensity results onto them using 'AGENT_CODE'



from catboost import CatBoostClassifier

# Define your training and validation data
X_train = base_data[variables]
X_test = val_data[variables]

# Initialize models with the provided best parameters
model_t1 = CatBoostClassifier(scale_pos_weight=3, learning_rate=0.05, l2_leaf_reg=7, iterations=200, depth=4, random_state=42, verbose=0)
model_t2 = CatBoostClassifier(scale_pos_weight=3, learning_rate=0.05, l2_leaf_reg=7, iterations=200, depth=4, random_state=42, verbose=0)
model_t3 = CatBoostClassifier(scale_pos_weight=3, learning_rate=0.05, l2_leaf_reg=7, iterations=200, depth=4, random_state=42, verbose=0)

# Fit the models
print("\n--- Training model for T+1 ---\n")
y_train_t1 = base_data['T+1']
model_t1.fit(X_train, y_train_t1)
t1_predictions = model_t1.predict_proba(X_test)[:, 1]  # Probability of class 1

print("\n--- Training model for T+2 ---\n")
y_train_t2 = base_data['T+2']
model_t2.fit(X_train, y_train_t2)
t2_predictions = model_t2.predict_proba(X_test)[:, 1]  # Probability of class 1

print("\n--- Training model for T+3 ---\n")
y_train_t3 = base_data['T+3']
model_t3.fit(X_train, y_train_t3)
t3_predictions = model_t3.predict_proba(X_test)[:, 1]  # Probability of class 1

# Assign the predictions to the validation data
val_data['T+1pred'] = t1_predictions
val_data['T+2pred'] = t2_predictions
val_data['T+3pred'] = t3_predictions
val_data['AGENT_CODE'] = val_agent_code

# Merge predictions with final_data using 'AGENT_CODE'
final_data = final_data.merge(val_data[['AGENT_CODE', 'T+1pred', 'T+2pred', 'T+3pred']], on='AGENT_CODE', how='left')

# Now final_data contains the predictions for T+1, T+2, and T+3



Hi In this code , can you store the three models at the end , X_train = base_data[variables]
X_test = val_data[variables]
# Function to train and evaluate a model
def train_evaluate_model(model, X_train, y_train, X_test, model_name):
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train the model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model
# Model training and tuning for each target
best_models = {}
for target in ['T+1', 'T+2', 'T+3']:
    print(f"\n--- Training models for {target} ---\n")
    y_train = base_data[target]
    # y_test = val_data[target]
    
    # CatBoost
    cat_params = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'scale_pos_weight': [1, 2, 3]  # Added to boost recall
    }
    cat = CatBoostClassifier(random_state=42, verbose=0)
    cat_cv = RandomizedSearchCV(cat, cat_params, n_iter=10, cv=3, scoring='recall', n_jobs=-1, random_state=42)
    cat_cv.fit(X_train, y_train)
    best_cat = train_evaluate_model(cat_cv.best_estimator_, X_train, y_train, X_test, "CatBoost")
    




from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV

# Define your training and validation data
X_train = base_data[variables]
X_test = val_data[variables]

# Function to train and evaluate a model
def train_evaluate_model(model, X_train, y_train, X_test):
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train the model
    model.fit(X_train_resampled, y_train_resampled)
    
    return model

# Initialize and train models for each target
best_models = {}
for target, model_name in zip(['T+1', 'T+2', 'T+3'], ['model_t1', 'model_t2', 'model_t3']):
    print(f"\n--- Training model for {target} ---\n")
    y_train = base_data[target]
    
    # Best parameters for each model
    best_params = {
        'scale_pos_weight': 3,
        'learning_rate': 0.05,
        'l2_leaf_reg': 7,
        'iterations': 200,
        'depth': 4
    }
    
    # Initialize the CatBoost model with the best parameters
    cat = CatBoostClassifier(**best_params, random_state=42, verbose=0)
    
    # Train and evaluate the model
    best_model = train_evaluate_model(cat, X_train, y_train, X_test)
    
    # Store the model
    best_models[model_name] = best_model

# Model predictions (optional)
t1_predictions = best_models['model_t1'].predict_proba(X_test)[:, 1]  # Probability of class 1
t2_predictions = best_models['model_t2'].predict_proba(X_test)[:, 1]  # Probability of class 1
t3_predictions = best_models['model_t3'].predict_proba(X_test)[:, 1]  # Probability of class 1

# Assign the predictions to the validation data
val_data['T+1pred'] = t1_predictions
val_data['T+2pred'] = t2_predictions
val_data['T+3pred'] = t3_predictions
val_data['AGENT_CODE'] = val_agent_code

# Merge predictions with final_data using 'AGENT_CODE'
final_data = final_data.merge(val_data[['AGENT_CODE', 'T+1pred', 'T+2pred', 'T+3pred']], on='AGENT_CODE', how='left')

# At this point, the best_models dictionary contains the trained models
# You can access them via best_models['model_t1'], best_models['model_t2'], best_models['model_t3']



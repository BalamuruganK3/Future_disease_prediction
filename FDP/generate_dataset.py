import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 10000

JOB_LIST = [
    'Software Engineer','Doctor','Teacher','Farmer','Driver',
    'Nurse','Office Worker','Construction Worker','Chef','Student',
    'Manager','Retired','Businessman','Security Guard','Factory Worker',
    'Homemaker','House Maid','Delivery Boy','Auto Driver','Shopkeeper',
    'Tailor','Electrician','Mechanic','Daily Wage Worker','Street Vendor'
]
BLOOD_GROUPS = ['A+','A-','B+','B-','AB+','AB-','O+','O-']

def generate():
    ages     = np.random.randint(18, 80, N)
    weights  = np.random.randint(45, 120, N)
    heights  = np.random.randint(150, 195, N)
    bgroups  = np.random.choice(BLOOD_GROUPS, N)
    jobs     = np.random.choice(JOB_LIST, N)
    sleep    = np.round(np.random.uniform(1.0, 10.0, N), 1)   # min 1 hr
    stress   = np.random.randint(1, 11, N)                     # min 1
    exercise = np.random.randint(0, 8, N)
    smoking  = np.random.choice([0,1], N, p=[0.7,0.3])
    alcohol  = np.random.choice([0,1], N, p=[0.65,0.35])
    fhd      = np.random.choice([0,1], N, p=[0.7,0.3])
    fhh      = np.random.choice([0,1], N, p=[0.75,0.25])
    sbp      = np.random.randint(90, 180, N)
    dbp      = np.random.randint(60, 120, N)
    gluc     = np.random.randint(70, 200, N)
    aqi      = np.random.choice([0,1,2,3,4,5], N, p=[0.15,0.25,0.25,0.18,0.12,0.05])

    h        = heights / 100.0
    bmis     = np.round(weights / (h * h), 2)

    hazard   = np.isin(jobs, ['Factory Worker','Construction Worker','Driver','Farmer','Auto Driver','Electrician','Mechanic','Daily Wage Worker'])
    sedent   = np.isin(jobs, ['Software Engineer','Office Worker','Manager','Businessman','Student','Shopkeeper'])

    diabetes = ((bmis>30)*2 + (gluc>125)*3 + (ages>40) + fhd + (exercise<2)) >= 4
    hyperten = ((sbp>140)*2 + (dbp>90)*2 + (stress>7) + (ages>45) + smoking + (bmis>27)) >= 4
    heart    = ((ages>50)*2 + (sbp>140) + smoking*2 + fhh*2 + (stress>6) + (bmis>30) + alcohol) >= 5
    thyroid  = ((stress>6) + (ages>30) + (sleep<6) + (np.random.random(N)<0.15)*2) >= 3
    sleepdis = ((sleep<6)*2 + (stress>7) + sedent + (ages>35)) >= 3
    obesity  = ((bmis>30)*3 + (exercise<2)*2 + (ages>35) + alcohol) >= 4
    occup    = (hazard*3 + (ages>40) + (exercise>5) + smoking) >= 3
    resp     = ((aqi>=2) + (aqi>=3) + (aqi>=4)*2 + smoking*2 + hazard + (ages>40) + (exercise<2)) >= 4

    jmap = {j:i for i,j in enumerate(JOB_LIST)}
    bmap = {b:i for i,b in enumerate(BLOOD_GROUPS)}

    df = pd.DataFrame({
        'age':ages,'weight':weights,'height':heights,'bmi':bmis,
        'blood_group':[bmap[b] for b in bgroups],
        'job':[jmap[j] for j in jobs],
        'sleep_hours':sleep,'stress_level':stress,
        'physical_activity_days':exercise,
        'smoking':smoking,'alcohol':alcohol,
        'family_history_diabetes':fhd,'family_history_heart':fhh,
        'systolic_bp':sbp,'diastolic_bp':dbp,'fasting_glucose':gluc,
        'aqi_category':aqi,
        'diabetes':diabetes.astype(int),'hypertension':hyperten.astype(int),
        'heart_disease':heart.astype(int),'hypothyroidism':thyroid.astype(int),
        'sleep_disorder':sleepdis.astype(int),'obesity_risk':obesity.astype(int),
        'occupational_disease':occup.astype(int),'respiratory_disease':resp.astype(int),
    })
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/health_dataset.csv', index=False)
    print(f"Dataset saved: {N} rows, {len(df.columns)} columns")
    for c in ['diabetes','hypertension','heart_disease','hypothyroidism',
              'sleep_disorder','obesity_risk','occupational_disease','respiratory_disease']:
        print(f"  {c}: {df[c].mean()*100:.1f}%")
    return df

if __name__ == '__main__':
    generate()

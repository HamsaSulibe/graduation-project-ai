import os

dirs = [
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\data\models',
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\data\datasources',
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\data\repositories',
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\logic\bloc',
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\presentation\widgets',
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f'Created: {d}')

import os

directories = [
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\data\models',
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\data\datasources',
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\data\repositories',
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\logic\bloc',
    r'c:\Users\lenovo\OneDrive\Desktop\graduation-\graduation-project-frontend\lib\features\assistant\presentation\widgets'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f'Created: {directory}')

print('All directories created successfully!')

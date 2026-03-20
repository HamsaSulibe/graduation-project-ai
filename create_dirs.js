const fs = require('fs');

const directories = [
    'c:\\Users\\lenovo\\OneDrive\\Desktop\\graduation-\\graduation-project-frontend\\lib\\features\\assistant\\data\\models',
    'c:\\Users\\lenovo\\OneDrive\\Desktop\\graduation-\\graduation-project-frontend\\lib\\features\\assistant\\data\\datasources',
    'c:\\Users\\lenovo\\OneDrive\\Desktop\\graduation-\\graduation-project-frontend\\lib\\features\\assistant\\data\\repositories',
    'c:\\Users\\lenovo\\OneDrive\\Desktop\\graduation-\\graduation-project-frontend\\lib\\features\\assistant\\logic\\bloc',
    'c:\\Users\\lenovo\\OneDrive\\Desktop\\graduation-\\graduation-project-frontend\\lib\\features\\assistant\\presentation\\widgets'
];

directories.forEach(dir => {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`Created: ${dir}`);
});

console.log('All directories created successfully!');

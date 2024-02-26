# Configuration 

> Instructions for setting your system at LCLS servers 
-----------------------------
### setting up GitHub repo
#### configure 
```bash
git config --global user.name fperakis
git config --global color.ui auto
git config --global user.email f.perakis@fysik.su.se
```

#### standard git commands
```bash
git clone https://github.com/fperakis/LCLS_L10010 .git
git status
git add [file]
git commit -m “[message]”
git push
git pull
```
note that to be able to push you will need to create a PAT (instead of password), I used the classic token
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token 

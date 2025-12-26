# UI/UX Improvements for Authentication System

## Username Input Field
- The textbox for username entry should be **disabled** while the system is scanning
- This prevents accidental modifications during the authentication process

## Username Matching Logic
- When the same username is entered, the system should recognize it as the same user
- Currently, duplicate users exist in the database despite identical usernames
- **TODO:** Implement username uniqueness constraint or merge duplicate user records

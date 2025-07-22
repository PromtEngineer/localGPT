# User Signup Process Documentation

## How Users Sign Up in LocalGPT

The multi-user LocalGPT system now supports user registration and authentication through a modal-based interface. Here's how the signup process works:

### Step-by-Step User Signup Flow

1. **Access the Authentication Modal**
   - Users click the "Sign In" button in the top right corner of the LocalGPT interface
   - This opens the authentication modal overlay

2. **Switch to Registration Mode**
   - In the authentication modal, users see a "Sign In" form by default
   - At the bottom of the modal, there's a "Don't have an account? Sign up" link
   - Clicking this link switches the modal to "Create Account" mode

3. **Fill Registration Form**
   - The registration form requires three fields:
     - **Username**: Unique identifier for the user
     - **Email**: User's email address
     - **Password**: Secure password for the account
   - All fields are required for successful registration

4. **Submit Registration**
   - Users click the "Create Account" button to submit their registration
   - The system validates the input and creates a new user account
   - Upon successful registration, the user is automatically logged in

5. **Post-Registration Experience**
   - After successful signup/login, users receive JWT tokens for API authentication
   - The UI updates to show the user's authenticated state
   - Users can now access protected features like creating indexes and chatting

### Technical Implementation

- **Frontend**: Authentication modal implemented in `src/components/ui/auth-modal.tsx`
- **Backend**: User management handled by `backend/auth.py` with JWT token authentication
- **Database**: User data stored in SQLite with proper isolation between users
- **API Integration**: All API calls include user authentication headers

### User Data Isolation

Each user's data is completely isolated:
- **Sessions**: Each chat session belongs to a specific user
- **Indexes**: Document indexes are user-specific with unique vector table names
- **Access Control**: Users can only access their own sessions and indexes

### Authentication Flow

1. User registers → Account created in database
2. User logs in → JWT token generated and stored
3. Protected API calls → Token validated on each request
4. User data → Filtered by authenticated user ID

This ensures complete multi-user support with proper data separation and security.

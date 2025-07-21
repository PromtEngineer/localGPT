import hashlib
import secrets
import jwt
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import wraps
import json

class AuthManager:
    def __init__(self, db_path: Optional[str] = None, secret_key: Optional[str] = None):
        if db_path is None:
            import os
            if os.path.exists("/app"):
                self.db_path = "/app/backend/chat_data.db"
            else:
                self.db_path = "backend/chat_data.db"
        else:
            self.db_path = db_path
            
        self.secret_key = secret_key or self._generate_secret_key()
        
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key for JWT tokens"""
        return secrets.token_urlsafe(32)
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against its hash"""
        try:
            salt, password_hash = stored_hash.split(':')
            return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
        except ValueError:
            return False
    
    def create_user(self, username: str, email: str, password: str) -> Optional[str]:
        """Create a new user account"""
        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(password)
        now = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO users (id, username, email, password_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, email, password_hash, now, now))
            conn.commit()
            conn.close()
            
            print(f"✅ Created user: {username} ({user_id[:8]}...)")
            return user_id
            
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                raise ValueError("Username already exists")
            elif "email" in str(e):
                raise ValueError("Email already exists")
            else:
                raise ValueError("User creation failed")
        except Exception as e:
            raise ValueError(f"Database error: {str(e)}")
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user and return user info if successful"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT id, username, email, password_hash, is_active, metadata
            FROM users
            WHERE username = ? AND is_active = 1
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user and self._verify_password(password, user['password_hash']):
            return {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'is_active': bool(user['is_active']),
                'metadata': json.loads(user['metadata'] or '{}')
            }
        
        return None
    
    def generate_token(self, user_id: str, expires_hours: int = 24) -> str:
        """Generate a JWT token for a user"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify a JWT token and return user_id if valid"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload.get('user_id')
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information by user_id"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT id, username, email, created_at, updated_at, is_active, metadata
            FROM users
            WHERE id = ? AND is_active = 1
        ''', (user_id,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'created_at': user['created_at'],
                'updated_at': user['updated_at'],
                'is_active': bool(user['is_active']),
                'metadata': json.loads(user['metadata'] or '{}')
            }
        
        return None
    
    def update_user(self, user_id: str, **updates) -> bool:
        """Update user information"""
        allowed_fields = ['username', 'email', 'metadata']
        update_fields = []
        values = []
        
        for field, value in updates.items():
            if field in allowed_fields:
                if field == 'metadata':
                    value = json.dumps(value)
                update_fields.append(f"{field} = ?")
                values.append(value)
        
        if not update_fields:
            return False
        
        values.append(datetime.now().isoformat())
        values.append(user_id)
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(f'''
                UPDATE users 
                SET {', '.join(update_fields)}, updated_at = ?
                WHERE id = ?
            ''', values)
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
        result = cursor.fetchone()
        
        if not result or not self._verify_password(old_password, result[0]):
            conn.close()
            return False
        
        new_hash = self._hash_password(new_password)
        conn.execute('''
            UPDATE users 
            SET password_hash = ?, updated_at = ?
            WHERE id = ?
        ''', (new_hash, datetime.now().isoformat(), user_id))
        conn.commit()
        conn.close()
        return True
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                UPDATE users 
                SET is_active = 0, updated_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), user_id))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

def require_auth(auth_manager: AuthManager):
    """Decorator to require authentication for API endpoints"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            auth_header = self.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                self.send_json_response({'error': 'Authentication required'}, status_code=401)
                return
            
            token = auth_header.split(' ')[1]
            user_id = auth_manager.verify_token(token)
            
            if not user_id:
                self.send_json_response({'error': 'Invalid or expired token'}, status_code=401)
                return
            
            self.current_user_id = user_id
            return func(self, *args, **kwargs)
        
        return wrapper
    return decorator

#!/usr/bin/env python3
"""
Test script for multi-user functionality in LocalGPT
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.database import ChatDatabase
from backend.auth import AuthManager
import json

def test_multiuser_functionality():
    """Test multi-user database operations"""
    print("🧪 Testing Multi-User Functionality")
    print("=" * 50)
    
    db = ChatDatabase()
    auth = AuthManager()
    
    try:
        print("\n1. Testing User Creation")
        user1_id = auth.create_user("alice", "alice@example.com", "password123")
        user2_id = auth.create_user("bob", "bob@example.com", "password456")
        print(f"✅ Created user Alice: {user1_id[:8]}...")
        print(f"✅ Created user Bob: {user2_id[:8]}...")
        
        print("\n2. Testing User Authentication")
        alice = auth.authenticate_user("alice", "password123")
        bob = auth.authenticate_user("bob", "password456")
        invalid = auth.authenticate_user("alice", "wrongpassword")
        
        assert alice is not None, "Alice authentication failed"
        assert bob is not None, "Bob authentication failed"
        assert invalid is None, "Invalid authentication should fail"
        print("✅ User authentication working correctly")
        
        print("\n3. Testing JWT Tokens")
        alice_token = auth.generate_token(alice['id'])
        bob_token = auth.generate_token(bob['id'])
        
        alice_verified = auth.verify_token(alice_token)
        bob_verified = auth.verify_token(bob_token)
        
        assert alice_verified == alice['id'], "Alice token verification failed"
        assert bob_verified == bob['id'], "Bob token verification failed"
        print("✅ JWT token generation and verification working")
        
        print("\n4. Testing User-Specific Sessions")
        alice_session = db.create_session("Alice's Chat", "llama3.2:latest", alice['id'])
        bob_session = db.create_session("Bob's Chat", "llama3.2:latest", bob['id'])
        
        alice_sessions = db.get_sessions(alice['id'])
        bob_sessions = db.get_sessions(bob['id'])
        
        assert len(alice_sessions) == 1, f"Alice should have 1 session, got {len(alice_sessions)}"
        assert len(bob_sessions) == 1, f"Bob should have 1 session, got {len(bob_sessions)}"
        assert alice_sessions[0]['id'] == alice_session, "Alice session mismatch"
        assert bob_sessions[0]['id'] == bob_session, "Bob session mismatch"
        print("✅ User-specific session isolation working")
        
        print("\n5. Testing Cross-User Session Access")
        alice_session_from_bob = db.get_session(alice_session, bob['id'])
        bob_session_from_alice = db.get_session(bob_session, alice['id'])
        
        assert alice_session_from_bob is None, "Bob should not access Alice's session"
        assert bob_session_from_alice is None, "Alice should not access Bob's session"
        print("✅ Cross-user session access properly blocked")
        
        print("\n6. Testing User-Specific Indexes")
        alice_index = db.create_index("Alice's Index", alice['id'], "Alice's documents")
        bob_index = db.create_index("Bob's Index", bob['id'], "Bob's documents")
        
        alice_indexes = db.list_indexes(alice['id'])
        bob_indexes = db.list_indexes(bob['id'])
        
        assert len(alice_indexes) == 1, f"Alice should have 1 index, got {len(alice_indexes)}"
        assert len(bob_indexes) == 1, f"Bob should have 1 index, got {len(bob_indexes)}"
        assert alice_indexes[0]['id'] == alice_index, "Alice index mismatch"
        assert bob_indexes[0]['id'] == bob_index, "Bob index mismatch"
        print("✅ User-specific index isolation working")
        
        print("\n7. Testing Vector Table Naming")
        alice_vector_table = alice_indexes[0]['vector_table_name']
        bob_vector_table = bob_indexes[0]['vector_table_name']
        
        assert alice['id'][:8] in alice_vector_table, "Alice's user ID should be in vector table name"
        assert bob['id'][:8] in bob_vector_table, "Bob's user ID should be in vector table name"
        assert alice_vector_table != bob_vector_table, "Vector table names should be different"
        print("✅ User-specific vector table naming working")
        
        print("\n" + "=" * 50)
        print("🎉 ALL MULTI-USER TESTS PASSED!")
        print("✅ User authentication and authorization working")
        print("✅ Data isolation between users working")
        print("✅ Session and index access control working")
        print("✅ Vector table isolation working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multiuser_functionality()
    sys.exit(0 if success else 1)

"""
Complete Implementation: Cryptographic Benchmark Suite for MongoDB
===============================================================
Enhanced version with comprehensive metrics, multiple data types,
and formatted output for research paper quality results.

Features:
- Symmetric encryption: AES-256-GCM, ChaCha20-Poly1305, AES-256-CBC, Blowfish
- Asymmetric key wrapping: RSA-2048/4096, ECC P-256/P-384
- INDIVIDUAL ENCRYPTION/DECRYPTION functions for both symmetric and asymmetric
- RSA data encryption (not just key wrapping)
- Hybrid encryption (Envelope pattern)
- Multiple data types: JSON, Binary files, Geospatial, Emails, UUIDs
- Bit-flip fault injection testing
- Memory safety analysis
- Comprehensive performance metrics with throughput calculations
- COMPLETE TABLE OUTPUT with ALL results
- PROFESSIONAL CHARTS AND FIGURES for research papers
- Formatted output for research papers
"""

import os
import sys
import time
import json
import hashlib
import secrets
import random
import gc
import tracemalloc
import tempfile
from typing import Dict, Tuple, Optional, Any, List, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from decimal import Decimal
from datetime import datetime
import struct
import warnings
import base64

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Cryptography libraries
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.hmac import HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

# Database libraries
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.collection import Collection
    from pymongo.errors import OperationFailure
    from bson import Binary, json_util
    import bson
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("Warning: PyMongo not installed. MongoDB integration will be skipped.")

# Benchmarking and utilities
import pytest
import pytest_benchmark
from pytest_benchmark.fixture import BenchmarkFixture
import numpy as np
from typing import Generator

# For better output formatting
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("Warning: tabulate not installed. Install with: pip install tabulate")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# For charts and figures
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Charts will be skipped. Install with: pip install matplotlib")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the benchmark suite."""
    # MongoDB connection
    MONGODB_URI: str = "mongodb://localhost:27017/"
    DATABASE_NAME: str = "crypto_benchmark"
    
    # Test data sizes (in bytes)
    PAYLOAD_SIZES: List[int] = field(default_factory=lambda: [64, 256, 1024, 4096, 16384, 65536])
    
    # Number of iterations for each test
    ITERATIONS: int = 50
    ASYMMETRIC_ITERATIONS: int = 20
    
    # Test data types
    DATA_TYPES: List[str] = field(default_factory=lambda: ['email', 'json_document', 'uuid', 'binary_file'])
    
    # Security levels
    SYMMETRIC_KEY_SIZES: Dict[str, int] = field(default_factory=lambda: {
        'AES-128': 16,
        'AES-256': 32,
        'ChaCha20': 32,
        'Blowfish': 16,
    })
    
    # RSA key sizes
    RSA_KEY_SIZES: List[int] = field(default_factory=lambda: [2048, 4096])
    
    # ECC curves
    ECC_CURVES: Dict[str, Any] = field(default_factory=lambda: {
        'P-256': ec.SECP256R1(),
        'P-384': ec.SECP384R1(),
    })

# Create a single config instance
config = Config()

# ============================================================================
# CORE CRYPTOGRAPHY IMPLEMENTATIONS
# ============================================================================

class CryptoProvider:
    """Base class for cryptographic operations."""
    
    def __init__(self, backend=default_backend()):
        self.backend = backend
    
    def generate_key(self) -> bytes:
        raise NotImplementedError
    
    def encrypt(self, plaintext: bytes, key: bytes, aad: bytes = b'') -> Dict[str, bytes]:
        raise NotImplementedError
    
    def decrypt(self, ciphertext_data: Dict[str, bytes], key: bytes) -> bytes:
        raise NotImplementedError
    
    def get_name(self) -> str:
        raise NotImplementedError
    
    def get_key_size(self) -> int:
        raise NotImplementedError
    
    def encrypt_simple(self, plaintext: bytes, key: bytes) -> bytes:
        """Simple encryption returning just ciphertext (for demos)."""
        result = self.encrypt(plaintext, key)
        return result['ciphertext']
    
    def decrypt_simple(self, ciphertext: bytes, key: bytes, iv: bytes = None) -> bytes:
        """Simple decryption with just ciphertext (for demos)."""
        ciphertext_data = {'ciphertext': ciphertext}
        if iv is not None:
            ciphertext_data['iv'] = iv
            ciphertext_data['nonce'] = iv
        return self.decrypt(ciphertext_data, key)

class AESGCMProvider(CryptoProvider):
    """AES-256-GCM implementation with authenticated encryption."""
    
    def generate_key(self) -> bytes:
        return os.urandom(32)
    
    def encrypt(self, plaintext: bytes, key: bytes, aad: bytes = b'') -> Dict[str, bytes]:
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
        return {
            'ciphertext': ciphertext,
            'nonce': nonce,
            'tag': ciphertext[-16:],
            'ciphertext_only': ciphertext[:-16]
        }
    
    def decrypt(self, ciphertext_data: Dict[str, bytes], key: bytes) -> bytes:
        aesgcm = AESGCM(key)
        nonce = ciphertext_data['nonce']
        ciphertext = ciphertext_data['ciphertext']
        return aesgcm.decrypt(nonce, ciphertext, b'')
    
    def get_name(self) -> str:
        return "AES-256-GCM"
    
    def get_key_size(self) -> int:
        return 256

class ChaCha20Provider(CryptoProvider):
    """ChaCha20-Poly1305 implementation."""
    
    def generate_key(self) -> bytes:
        return os.urandom(32)
    
    def encrypt(self, plaintext: bytes, key: bytes, aad: bytes = b'') -> Dict[str, bytes]:
        chacha = ChaCha20Poly1305(key)
        nonce = os.urandom(12)
        ciphertext = chacha.encrypt(nonce, plaintext, aad)
        return {
            'ciphertext': ciphertext,
            'nonce': nonce,
            'tag': ciphertext[-16:],
            'ciphertext_only': ciphertext[:-16]
        }
    
    def decrypt(self, ciphertext_data: Dict[str, bytes], key: bytes) -> bytes:
        chacha = ChaCha20Poly1305(key)
        nonce = ciphertext_data['nonce']
        ciphertext = ciphertext_data['ciphertext']
        return chacha.decrypt(nonce, ciphertext, b'')
    
    def get_name(self) -> str:
        return "ChaCha20-Poly1305"
    
    def get_key_size(self) -> int:
        return 256

class AESCBCProvider(CryptoProvider):
    """AES-256-CBC implementation with PKCS7 padding."""
    
    def generate_key(self) -> bytes:
        return os.urandom(32)
    
    def encrypt(self, plaintext: bytes, key: bytes, aad: bytes = b'') -> Dict[str, bytes]:
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return {
            'ciphertext': ciphertext,
            'iv': iv,
        }
    
    def decrypt(self, ciphertext_data: Dict[str, bytes], key: bytes) -> bytes:
        iv = ciphertext_data['iv']
        ciphertext = ciphertext_data['ciphertext']
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
        return plaintext
    
    def get_name(self) -> str:
        return "AES-256-CBC"
    
    def get_key_size(self) -> int:
        return 256

class BlowfishProvider(CryptoProvider):
    """Blowfish encryption (for comparison purposes)."""
    
    def generate_key(self) -> bytes:
        return os.urandom(16)
    
    def encrypt(self, plaintext: bytes, key: bytes, aad: bytes = b'') -> Dict[str, bytes]:
        iv = os.urandom(8)
        cipher = Cipher(algorithms.Blowfish(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        padder = padding.PKCS7(algorithms.Blowfish.block_size).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return {
            'ciphertext': ciphertext,
            'iv': iv,
        }
    
    def decrypt(self, ciphertext_data: Dict[str, bytes], key: bytes) -> bytes:
        iv = ciphertext_data['iv']
        ciphertext = ciphertext_data['ciphertext']
        cipher = Cipher(algorithms.Blowfish(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        unpadder = padding.PKCS7(algorithms.Blowfish.block_size).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
        return plaintext
    
    def get_name(self) -> str:
        return "Blowfish-CBC"
    
    def get_key_size(self) -> int:
        return 128

# ============================================================================
# INDIVIDUAL SYMMETRIC ENCRYPTION/DECRYPTION FUNCTIONS
# ============================================================================

class SymmetricCrypto:
    """Individual symmetric encryption/decryption functions for demo and testing."""
    
    @staticmethod
    def aes_gcm_encrypt(plaintext: bytes, key: bytes) -> Dict[str, bytes]:
        provider = AESGCMProvider()
        return provider.encrypt(plaintext, key)
    
    @staticmethod
    def aes_gcm_decrypt(ciphertext_data: Dict[str, bytes], key: bytes) -> bytes:
        provider = AESGCMProvider()
        return provider.decrypt(ciphertext_data, key)
    
    @staticmethod
    def chacha20_encrypt(plaintext: bytes, key: bytes) -> Dict[str, bytes]:
        provider = ChaCha20Provider()
        return provider.encrypt(plaintext, key)
    
    @staticmethod
    def chacha20_decrypt(ciphertext_data: Dict[str, bytes], key: bytes) -> bytes:
        provider = ChaCha20Provider()
        return provider.decrypt(ciphertext_data, key)
    
    @staticmethod
    def aes_cbc_encrypt(plaintext: bytes, key: bytes) -> Dict[str, bytes]:
        provider = AESCBCProvider()
        return provider.encrypt(plaintext, key)
    
    @staticmethod
    def aes_cbc_decrypt(ciphertext_data: Dict[str, bytes], key: bytes) -> bytes:
        provider = AESCBCProvider()
        return provider.decrypt(ciphertext_data, key)
    
    @staticmethod
    def blowfish_encrypt(plaintext: bytes, key: bytes) -> Dict[str, bytes]:
        provider = BlowfishProvider()
        return provider.encrypt(plaintext, key)
    
    @staticmethod
    def blowfish_decrypt(ciphertext_data: Dict[str, bytes], key: bytes) -> bytes:
        provider = BlowfishProvider()
        return provider.decrypt(ciphertext_data, key)

# ============================================================================
# INDIVIDUAL ASYMMETRIC ENCRYPTION/DECRYPTION FUNCTIONS
# ============================================================================

class AsymmetricCrypto:
    """Individual asymmetric encryption/decryption functions."""
    
    @staticmethod
    def generate_rsa_keypair(key_size: int = 2048) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        return private_key, private_key.public_key()
    
    @staticmethod
    def rsa_encrypt_data(plaintext: bytes, public_key: rsa.RSAPublicKey) -> bytes:
        try:
            ciphertext = public_key.encrypt(
                plaintext,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return ciphertext
        except Exception as e:
            raise ValueError(f"RSA encryption failed: {e}. Data too large for RSA key size.")
    
    @staticmethod
    def rsa_decrypt_data(ciphertext: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        plaintext = private_key.decrypt(
            ciphertext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext
    
    @staticmethod
    def generate_ecc_keypair(curve: ec.EllipticCurve = ec.SECP256R1()) -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
        private_key = ec.generate_private_key(curve, default_backend())
        return private_key, private_key.public_key()
    
    @staticmethod
    def ecc_encrypt_data(plaintext: bytes, public_key: ec.EllipticCurvePublicKey) -> bytes:
        ephemeral_private = ec.generate_private_key(public_key.curve, default_backend())
        ephemeral_public = ephemeral_private.public_key()
        
        shared_secret = ephemeral_private.exchange(ec.ECDH(), public_key)
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'ECC-data-encryption',
            backend=default_backend()
        )
        encryption_key = hkdf.derive(shared_secret)
        
        aesgcm = AESGCM(encryption_key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext, b'')
        
        ephemeral_public_bytes = ephemeral_public.public_bytes(
            encoding=Encoding.X962,
            format=PublicFormat.UncompressedPoint
        )
        
        return ephemeral_public_bytes + nonce + ciphertext
    
    @staticmethod
    def ecc_decrypt_data(ciphertext_data: bytes, private_key: ec.EllipticCurvePrivateKey) -> bytes:
        curve = private_key.curve
        point_size = (curve.key_size + 7) // 8 * 2 + 1
        
        ephemeral_public_bytes = ciphertext_data[:point_size]
        nonce = ciphertext_data[point_size:point_size+12]
        ciphertext = ciphertext_data[point_size+12:]
        
        ephemeral_public = ec.EllipticCurvePublicKey.from_encoded_point(curve, ephemeral_public_bytes)
        
        shared_secret = private_key.exchange(ec.ECDH(), ephemeral_public)
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'ECC-data-encryption',
            backend=default_backend()
        )
        encryption_key = hkdf.derive(shared_secret)
        
        aesgcm = AESGCM(encryption_key)
        return aesgcm.decrypt(nonce, ciphertext, b'')
    
    @staticmethod
    def rsa_encrypt_with_key_size(plaintext: bytes, key_size: int = 2048) -> Dict[str, Any]:
        private_key, public_key = AsymmetricCrypto.generate_rsa_keypair(key_size)
        ciphertext = AsymmetricCrypto.rsa_encrypt_data(plaintext, public_key)
        return {
            'ciphertext': ciphertext,
            'private_key': private_key,
            'public_key': public_key,
            'key_size': key_size
        }
    
    @staticmethod
    def rsa_decrypt_with_key(ciphertext: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        return AsymmetricCrypto.rsa_decrypt_data(ciphertext, private_key)

# ============================================================================
# HYBRID ENCRYPTION SYSTEM (Envelope Encryption)
# ============================================================================

class HybridEncryption:
    """Implements the Envelope Encryption pattern."""
    
    def __init__(self, symmetric_provider: CryptoProvider):
        self.symmetric = symmetric_provider
        self.asymmetric_wrapper = AsymmetricKeyWrapper()
    
    def encrypt_with_key_wrapping(
        self,
        plaintext: bytes,
        public_key: Union[rsa.RSAPublicKey, ec.EllipticCurvePublicKey],
        key_type: str = 'rsa'
    ) -> Dict[str, Any]:
        dek = self.symmetric.generate_key()
        ciphertext_data = self.symmetric.encrypt(plaintext, dek)
        
        if key_type == 'rsa':
            wrapped_dek = self.asymmetric_wrapper.wrap_key_rsa(dek, public_key)
        else:
            wrapped_dek = self.asymmetric_wrapper.wrap_key_ecc(dek, public_key)
        
        return {
            'wrapped_dek': wrapped_dek,
            'ciphertext_data': ciphertext_data,
            'key_type': key_type,
        }
    
    def decrypt_with_key_unwrapping(
        self,
        encrypted_data: Dict[str, Any],
        private_key: Union[rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey]
    ) -> bytes:
        wrapped_dek = encrypted_data['wrapped_dek']
        ciphertext_data = encrypted_data['ciphertext_data']
        key_type = encrypted_data.get('key_type', 'rsa')
        
        if key_type == 'rsa':
            dek = self.asymmetric_wrapper.unwrap_key_rsa(wrapped_dek, private_key)
        else:
            dek = self.asymmetric_wrapper.unwrap_key_ecc(wrapped_dek, private_key)
        
        return self.symmetric.decrypt(ciphertext_data, dek)

# ============================================================================
# ASYMMETRIC KEY WRAPPER (for key wrapping)
# ============================================================================

class AsymmetricKeyWrapper:
    """Handles asymmetric key wrapping for hybrid encryption."""
    
    def __init__(self, backend=default_backend()):
        self.backend = backend
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        return private_key, private_key.public_key()
    
    def generate_ecc_keypair(self, curve: ec.EllipticCurve = ec.SECP256R1()) -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
        private_key = ec.generate_private_key(curve, self.backend)
        return private_key, private_key.public_key()
    
    def wrap_key_rsa(self, key_to_wrap: bytes, public_key: rsa.RSAPublicKey) -> bytes:
        return public_key.encrypt(
            key_to_wrap,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def unwrap_key_rsa(self, wrapped_key: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        return private_key.decrypt(
            wrapped_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def wrap_key_ecc(self, key_to_wrap: bytes, public_key: ec.EllipticCurvePublicKey) -> bytes:
        ephemeral_private = ec.generate_private_key(public_key.curve, self.backend)
        ephemeral_public = ephemeral_private.public_key()
        
        shared_secret = ephemeral_private.exchange(ec.ECDH(), public_key)
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'ECC-key-wrap',
            backend=self.backend
        )
        encryption_key = hkdf.derive(shared_secret)
        
        aesgcm = AESGCM(encryption_key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, key_to_wrap, b'')
        
        ephemeral_public_bytes = ephemeral_public.public_bytes(
            encoding=Encoding.X962,
            format=PublicFormat.UncompressedPoint
        )
        
        return ephemeral_public_bytes + nonce + ciphertext
    
    def unwrap_key_ecc(self, wrapped_key_data: bytes, private_key: ec.EllipticCurvePrivateKey) -> bytes:
        curve = private_key.curve
        point_size = (curve.key_size + 7) // 8 * 2 + 1
        ephemeral_public_bytes = wrapped_key_data[:point_size]
        nonce = wrapped_key_data[point_size:point_size+12]
        ciphertext = wrapped_key_data[point_size+12:]
        
        ephemeral_public = ec.EllipticCurvePublicKey.from_encoded_point(curve, ephemeral_public_bytes)
        
        shared_secret = private_key.exchange(ec.ECDH(), ephemeral_public)
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'ECC-key-wrap',
            backend=self.backend
        )
        encryption_key = hkdf.derive(shared_secret)
        
        aesgcm = AESGCM(encryption_key)
        return aesgcm.decrypt(nonce, ciphertext, b'')

# ============================================================================
# FAULT INJECTION AND RESILIENCE TESTING
# ============================================================================

class FaultInjector:
    """Injects faults into ciphertext to test resilience."""
    
    @staticmethod
    def flip_bit(data: bytes, position: int) -> bytes:
        if position >= len(data) * 8:
            raise ValueError("Bit position out of range")
        
        byte_pos = position // 8
        bit_pos = position % 8
        byte_value = data[byte_pos]
        flipped_byte = byte_value ^ (1 << bit_pos)
        
        return data[:byte_pos] + bytes([flipped_byte]) + data[byte_pos+1:]
    
    @staticmethod
    def test_bit_flip_resilience(
        provider: CryptoProvider,
        plaintext: bytes,
        key: bytes,
        num_flips: int = 5,
        max_positions: int = 20
    ) -> Dict[str, Any]:
        encrypted_data = provider.encrypt(plaintext, key)
        ciphertext = encrypted_data['ciphertext']
        
        results = []
        if len(ciphertext) * 8 > 0:
            test_positions = random.sample(range(len(ciphertext) * 8), 
                                          min(max_positions, len(ciphertext) * 8))
        else:
            test_positions = []
        
        for pos in test_positions[:num_flips]:
            corrupted_ciphertext = FaultInjector.flip_bit(ciphertext, pos)
            
            try:
                if provider.get_name() in ['AES-256-GCM', 'ChaCha20-Poly1305']:
                    corrupted_data = encrypted_data.copy()
                    corrupted_data['ciphertext'] = corrupted_ciphertext
                    decrypted = provider.decrypt(corrupted_data, key)
                    
                    min_len = min(len(plaintext), len(decrypted))
                    damaged = sum(1 for i in range(min_len) if plaintext[i] != decrypted[i])
                    damage_ratio = damaged / len(plaintext) if len(plaintext) > 0 else 0
                    
                    results.append({
                        'position': pos,
                        'success': True,
                        'damaged_bytes': damaged,
                        'damage_ratio': damage_ratio
                    })
                else:
                    corrupted_data = encrypted_data.copy()
                    corrupted_data['ciphertext'] = corrupted_ciphertext
                    decrypted = provider.decrypt(corrupted_data, key)
                    
                    min_len = min(len(plaintext), len(decrypted))
                    damaged = sum(1 for i in range(min_len) if plaintext[i] != decrypted[i])
                    damage_ratio = damaged / len(plaintext) if len(plaintext) > 0 else 0
                    
                    results.append({
                        'position': pos,
                        'success': True,
                        'damaged_bytes': damaged,
                        'damage_ratio': damage_ratio
                    })
                    
            except Exception as e:
                results.append({
                    'position': pos,
                    'success': False,
                    'error': str(e),
                    'damaged_bytes': len(plaintext),
                    'damage_ratio': 1.0
                })
        
        if not results:
            return {
                'algorithm': provider.get_name(),
                'plaintext_size': len(plaintext),
                'damage_ratio': 0,
                'decryption_success_rate': 0,
                'avg_damaged_bytes': 0
            }
        
        successful = sum(1 for r in results if r['success'])
        avg_damage = np.mean([r['damage_ratio'] for r in results]) if results else 0
        avg_damaged_bytes = np.mean([r['damaged_bytes'] for r in results]) if results else 0
        
        return {
            'algorithm': provider.get_name(),
            'plaintext_size': len(plaintext),
            'ciphertext_size': len(ciphertext),
            'damage_ratio': avg_damage,
            'avg_damaged_bytes': avg_damaged_bytes,
            'decryption_success_rate': successful / len(results) if results else 0,
            'sample_results': results[:5]
        }

# ============================================================================
# MEMORY SAFETY TESTING
# ============================================================================

class MemorySafetyTester:
    """Tests memory safety and cleanup of cryptographic material."""
    
    @staticmethod
    def test_memory_cleanup(
        provider: CryptoProvider,
        plaintext: bytes,
        key: bytes
    ) -> Dict[str, Any]:
        tracemalloc.start()
        
        snapshot1 = tracemalloc.take_snapshot()
        initial_memory = sum(stat.size for stat in snapshot1.statistics('filename'))
        
        encrypted_data = provider.encrypt(plaintext, key)
        gc.collect()
        
        snapshot2 = tracemalloc.take_snapshot()
        memory_after_encrypt = sum(stat.size for stat in snapshot2.statistics('filename'))
        
        decrypted = provider.decrypt(encrypted_data, key)
        gc.collect()
        
        snapshot3 = tracemalloc.take_snapshot()
        memory_final = sum(stat.size for stat in snapshot3.statistics('filename'))
        
        tracemalloc.stop()
        
        return {
            'algorithm': provider.get_name(),
            'plaintext_size': len(plaintext),
            'initial_memory': initial_memory,
            'memory_after_encrypt': memory_after_encrypt,
            'memory_final': memory_final,
            'memory_leak_suspected': memory_final > memory_after_encrypt + 1024,
            'memory_change_kb': (memory_final - memory_after_encrypt) / 1024,
        }

# ============================================================================
# BENCHMARK DATA GENERATORS
# ============================================================================

class DataGenerator:
    """Generates realistic test data for benchmarking."""
    
    @staticmethod
    def generate_email() -> str:
        domains = ['example.com', 'test.org', 'demo.net', 'company.io']
        names = ['john', 'jane', 'alice', 'bob', 'charlie', 'diana']
        return f"{random.choice(names)}{random.randint(1, 999)}@{random.choice(domains)}"
    
    @staticmethod
    def generate_uuid() -> str:
        return str(secrets.token_hex(16))
    
    @staticmethod
    def generate_geospatial() -> Dict[str, float]:
        return {
            'type': 'Point',
            'coordinates': [
                random.uniform(-180, 180),
                random.uniform(-90, 90)
            ]
        }
    
    @staticmethod
    def generate_json_document(size: int = 1024) -> Dict[str, Any]:
        doc = {
            'id': secrets.token_hex(8),
            'timestamp': time.time(),
            'name': f"User_{random.randint(1, 1000)}",
            'email': DataGenerator.generate_email(),
            'age': random.randint(18, 80),
            'address': {
                'street': f"{random.randint(1, 999)} Main St",
                'city': random.choice(['New York', 'London', 'Tokyo', 'Paris']),
                'country': random.choice(['USA', 'UK', 'Japan', 'France']),
                'zip': f"{random.randint(10000, 99999)}"
            },
            'metadata': {
                'created_at': time.time() - random.uniform(0, 86400),
                'updated_at': time.time(),
                'version': random.randint(1, 10),
                'active': random.choice([True, False])
            },
            'tags': random.sample(['premium', 'verified', 'active', 'trial'], random.randint(1, 3)),
            'score': random.uniform(0, 100),
            'location': DataGenerator.generate_geospatial()
        }
        
        current_size = len(json.dumps(doc).encode('utf-8'))
        while current_size < size:
            doc[f'extra_{random.randint(1, 100)}'] = secrets.token_hex(16)
            current_size = len(json.dumps(doc).encode('utf-8'))
        
        return doc
    
    @staticmethod
    def generate_binary_file(size: int = 1024) -> bytes:
        """Generate binary data simulating a file."""
        data = bytearray()
        
        # Add some header
        data.extend(b'FILE')
        data.extend(struct.pack('<I', size))
        
        # Add some structured data
        for i in range(size - 8):
            if i % 100 < 80:
                data.append((i * 7 + 13) % 256)
            else:
                data.append(random.randint(0, 255))
        
        return bytes(data[:size])
    
    @staticmethod
    def generate_payload_by_type(data_type: str, size: int = 1024) -> bytes:
        """Generate payload by data type."""
        if data_type == 'email':
            return DataGenerator.generate_email().encode('utf-8')
        elif data_type == 'uuid':
            return DataGenerator.generate_uuid().encode('utf-8')
        elif data_type == 'geospatial':
            return json.dumps(DataGenerator.generate_geospatial()).encode('utf-8')
        elif data_type == 'json_document':
            return json.dumps(DataGenerator.generate_json_document(size)).encode('utf-8')
        elif data_type == 'binary_file':
            return DataGenerator.generate_binary_file(size)
        else:
            return os.urandom(size)

# ============================================================================
# MONGODB OPERATIONS (Conditional)
# ============================================================================

if MONGO_AVAILABLE:
    class EncryptedMongoDB:
        """Handles encrypted operations with MongoDB."""
        
        def __init__(self, uri: str = config.MONGODB_URI, db_name: str = config.DATABASE_NAME):
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.encrypted_collections = {}
            
        def get_collection(self, name: str) -> Collection:
            if name not in self.encrypted_collections:
                self.encrypted_collections[name] = self.db[name]
            return self.encrypted_collections[name]
        
        def insert_encrypted_document(
            self,
            collection: Collection,
            document: Dict[str, Any],
            encryption_keys: Dict[str, bytes],
            algorithm_mapping: Dict[str, CryptoProvider],
            hybrid_public_key: Optional[Union[rsa.RSAPublicKey, ec.EllipticCurvePublicKey]] = None,
            use_hybrid: bool = False
        ) -> str:
            encrypted_doc = {}
            
            for field, value in document.items():
                if field in encryption_keys:
                    plaintext = json.dumps(value).encode('utf-8')
                    key = encryption_keys[field]
                    provider = algorithm_mapping.get(field)
                    
                    if provider:
                        if use_hybrid and hybrid_public_key:
                            hybrid = HybridEncryption(provider)
                            encrypted_data = hybrid.encrypt_with_key_wrapping(
                                plaintext,
                                hybrid_public_key,
                                key_type='rsa'
                            )
                            encrypted_doc[field] = {
                                '__encrypted__': True,
                                '__data__': Binary(encrypted_data['ciphertext_data']['ciphertext']),
                                '__iv__': Binary(encrypted_data['ciphertext_data'].get('iv', b'')),
                                '__nonce__': Binary(encrypted_data['ciphertext_data'].get('nonce', b'')),
                                '__wrapped_dek__': Binary(encrypted_data['wrapped_dek']),
                                '__key_type__': encrypted_data['key_type'],
                            }
                        else:
                            encrypted_data = provider.encrypt(plaintext, key)
                            encrypted_doc[field] = {
                                '__encrypted__': True,
                                '__data__': Binary(encrypted_data['ciphertext']),
                                '__iv__': Binary(encrypted_data.get('iv', b'')),
                                '__nonce__': Binary(encrypted_data.get('nonce', b'')),
                            }
                    else:
                        encrypted_doc[field] = value
                else:
                    encrypted_doc[field] = value
            
            encrypted_doc['__encryption_metadata__'] = {
                'version': '1.0',
                'timestamp': time.time(),
            }
            
            result = collection.insert_one(encrypted_doc)
            return str(result.inserted_id)

# ============================================================================
# DEMO FUNCTIONS FOR INDIVIDUAL ENCRYPTION/DECRYPTION
# ============================================================================

def demo_individual_symmetric_encryption():
    """Demo individual symmetric encryption/decryption functions."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL SYMMETRIC ENCRYPTION/DECRYPTION DEMO")
    print("=" * 80)
    
    test_data = b"Hello, this is a secret message for symmetric encryption testing!"
    
    print(f"\nOriginal Data: {test_data.decode('utf-8')}")
    print(f"Data Size: {len(test_data)} bytes")
    
    # AES-GCM
    print("\n" + "-" * 40)
    print("1. AES-256-GCM")
    key = os.urandom(32)
    print(f"  Key: {base64.b64encode(key).decode()[:16]}...")
    
    encrypted = SymmetricCrypto.aes_gcm_encrypt(test_data, key)
    decrypted = SymmetricCrypto.aes_gcm_decrypt(encrypted, key)
    
    print(f"  Ciphertext Size: {len(encrypted['ciphertext'])} bytes")
    print(f"  Nonce: {base64.b64encode(encrypted['nonce']).decode()[:12]}...")
    print(f"  Decrypted: {decrypted.decode('utf-8')}")
    print(f"  ✓ Success: {decrypted == test_data}")
    
    # ChaCha20
    print("\n" + "-" * 40)
    print("2. ChaCha20-Poly1305")
    key = os.urandom(32)
    print(f"  Key: {base64.b64encode(key).decode()[:16]}...")
    
    encrypted = SymmetricCrypto.chacha20_encrypt(test_data, key)
    decrypted = SymmetricCrypto.chacha20_decrypt(encrypted, key)
    
    print(f"  Ciphertext Size: {len(encrypted['ciphertext'])} bytes")
    print(f"  Nonce: {base64.b64encode(encrypted['nonce']).decode()[:12]}...")
    print(f"  Decrypted: {decrypted.decode('utf-8')}")
    print(f"  ✓ Success: {decrypted == test_data}")
    
    # AES-CBC
    print("\n" + "-" * 40)
    print("3. AES-256-CBC")
    key = os.urandom(32)
    print(f"  Key: {base64.b64encode(key).decode()[:16]}...")
    
    encrypted = SymmetricCrypto.aes_cbc_encrypt(test_data, key)
    decrypted = SymmetricCrypto.aes_cbc_decrypt(encrypted, key)
    
    print(f"  Ciphertext Size: {len(encrypted['ciphertext'])} bytes")
    print(f"  IV: {base64.b64encode(encrypted['iv']).decode()[:12]}...")
    print(f"  Decrypted: {decrypted.decode('utf-8')}")
    print(f"  ✓ Success: {decrypted == test_data}")
    
    # Blowfish
    print("\n" + "-" * 40)
    print("4. Blowfish-CBC")
    key = os.urandom(16)
    print(f"  Key: {base64.b64encode(key).decode()[:16]}...")
    
    encrypted = SymmetricCrypto.blowfish_encrypt(test_data, key)
    decrypted = SymmetricCrypto.blowfish_decrypt(encrypted, key)
    
    print(f"  Ciphertext Size: {len(encrypted['ciphertext'])} bytes")
    print(f"  IV: {base64.b64encode(encrypted['iv']).decode()[:12]}...")
    print(f"  Decrypted: {decrypted.decode('utf-8')}")
    print(f"  ✓ Success: {decrypted == test_data}")

def demo_individual_asymmetric_encryption():
    """Demo individual asymmetric encryption/decryption functions."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL ASYMMETRIC ENCRYPTION/DECRYPTION DEMO")
    print("=" * 80)
    
    test_data = b"RSA secret message!"
    
    print(f"\nOriginal Data: {test_data.decode('utf-8')}")
    print(f"Data Size: {len(test_data)} bytes")
    
    # RSA 2048
    print("\n" + "-" * 40)
    print("1. RSA-2048 (Direct Data Encryption)")
    print("  Note: RSA can only encrypt small data (max ~245 bytes for 2048-bit key)")
    
    private_key, public_key = AsymmetricCrypto.generate_rsa_keypair(2048)
    print(f"  Public Key Size: {public_key.key_size} bits")
    print(f"  Private Key Size: {private_key.key_size} bits")
    
    encrypted = AsymmetricCrypto.rsa_encrypt_data(test_data, public_key)
    decrypted = AsymmetricCrypto.rsa_decrypt_data(encrypted, private_key)
    
    print(f"  Ciphertext Size: {len(encrypted)} bytes")
    print(f"  Decrypted: {decrypted.decode('utf-8')}")
    print(f"  ✓ Success: {decrypted == test_data}")
    
    # RSA 4096
    print("\n" + "-" * 40)
    print("2. RSA-4096 (Direct Data Encryption)")
    
    private_key, public_key = AsymmetricCrypto.generate_rsa_keypair(4096)
    print(f"  Public Key Size: {public_key.key_size} bits")
    print(f"  Private Key Size: {private_key.key_size} bits")
    
    encrypted = AsymmetricCrypto.rsa_encrypt_data(test_data, public_key)
    decrypted = AsymmetricCrypto.rsa_decrypt_data(encrypted, private_key)
    
    print(f"  Ciphertext Size: {len(encrypted)} bytes")
    print(f"  Decrypted: {decrypted.decode('utf-8')}")
    print(f"  ✓ Success: {decrypted == test_data}")
    
    # ECC P-256
    print("\n" + "-" * 40)
    print("3. ECC P-256 (ECIES - Hybrid Encryption)")
    
    private_key, public_key = AsymmetricCrypto.generate_ecc_keypair(ec.SECP256R1())
    print(f"  Curve: P-256")
    
    encrypted = AsymmetricCrypto.ecc_encrypt_data(test_data, public_key)
    decrypted = AsymmetricCrypto.ecc_decrypt_data(encrypted, private_key)
    
    print(f"  Ciphertext Size: {len(encrypted)} bytes")
    print(f"  Decrypted: {decrypted.decode('utf-8')}")
    print(f"  ✓ Success: {decrypted == test_data}")
    
    # ECC P-384
    print("\n" + "-" * 40)
    print("4. ECC P-384 (ECIES - Hybrid Encryption)")
    
    private_key, public_key = AsymmetricCrypto.generate_ecc_keypair(ec.SECP384R1())
    print(f"  Curve: P-384")
    
    encrypted = AsymmetricCrypto.ecc_encrypt_data(test_data, public_key)
    decrypted = AsymmetricCrypto.ecc_decrypt_data(encrypted, private_key)
    
    print(f"  Ciphertext Size: {len(encrypted)} bytes")
    print(f"  Decrypted: {decrypted.decode('utf-8')}")
    print(f"  ✓ Success: {decrypted == test_data}")

def demo_hybrid_encryption():
    """Demo hybrid encryption (envelope pattern)."""
    print("\n" + "=" * 80)
    print("HYBRID ENCRYPTION (ENVELOPE PATTERN) DEMO")
    print("=" * 80)
    
    test_data = DataGenerator.generate_json_document(2048)
    plaintext = json.dumps(test_data).encode('utf-8')
    
    print(f"\nOriginal Data Size: {len(plaintext)} bytes")
    print(f"Original Data Preview: {plaintext[:50]}...")
    
    print("\n" + "-" * 40)
    print("1. RSA-2048 + AES-256-GCM (Envelope Encryption)")
    
    private_key, public_key = AsymmetricCrypto.generate_rsa_keypair(2048)
    print(f"  RSA Key Size: {public_key.key_size} bits")
    
    symmetric_provider = AESGCMProvider()
    hybrid = HybridEncryption(symmetric_provider)
    
    start_time = time.perf_counter_ns()
    encrypted_data = hybrid.encrypt_with_key_wrapping(plaintext, public_key, 'rsa')
    encrypt_time = (time.perf_counter_ns() - start_time) / 1000
    
    print(f"  Encrypt Time: {encrypt_time:.2f} μs")
    print(f"  Ciphertext Size: {len(encrypted_data['ciphertext_data']['ciphertext'])} bytes")
    print(f"  Wrapped DEK Size: {len(encrypted_data['wrapped_dek'])} bytes")
    print(f"  Total Size: {len(encrypted_data['ciphertext_data']['ciphertext']) + len(encrypted_data['wrapped_dek'])} bytes")
    
    start_time = time.perf_counter_ns()
    decrypted = hybrid.decrypt_with_key_unwrapping(encrypted_data, private_key)
    decrypt_time = (time.perf_counter_ns() - start_time) / 1000
    
    print(f"  Decrypt Time: {decrypt_time:.2f} μs")
    print(f"  ✓ Success: {decrypted == plaintext}")
    
    print("\n" + "-" * 40)
    print("2. ECC P-256 + ChaCha20 (Envelope Encryption)")
    
    private_key, public_key = AsymmetricCrypto.generate_ecc_keypair(ec.SECP256R1())
    print(f"  ECC Curve: P-256")
    
    symmetric_provider = ChaCha20Provider()
    hybrid = HybridEncryption(symmetric_provider)
    
    start_time = time.perf_counter_ns()
    encrypted_data = hybrid.encrypt_with_key_wrapping(plaintext, public_key, 'ecc')
    encrypt_time = (time.perf_counter_ns() - start_time) / 1000
    
    print(f"  Encrypt Time: {encrypt_time:.2f} μs")
    print(f"  Ciphertext Size: {len(encrypted_data['ciphertext_data']['ciphertext'])} bytes")
    print(f"  Wrapped DEK Size: {len(encrypted_data['wrapped_dek'])} bytes")
    print(f"  Total Size: {len(encrypted_data['ciphertext_data']['ciphertext']) + len(encrypted_data['wrapped_dek'])} bytes")
    
    start_time = time.perf_counter_ns()
    decrypted = hybrid.decrypt_with_key_unwrapping(encrypted_data, private_key)
    decrypt_time = (time.perf_counter_ns() - start_time) / 1000
    
    print(f"  Decrypt Time: {decrypt_time:.2f} μs")
    print(f"  ✓ Success: {decrypted == plaintext}")

# ============================================================================
# CHART GENERATION FUNCTIONS
# ============================================================================

class ChartGenerator:
    """Generate professional charts for research paper."""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
        self.chart_dir = f"charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if MATPLOTLIB_AVAILABLE:
            os.makedirs(self.chart_dir, exist_ok=True)
            # Set style for publication-quality charts
            plt.style.use('seaborn-v0_8-darkgrid')
            plt.rcParams['font.size'] = 10
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['figure.figsize'] = (10, 6)
    
    def generate_all_charts(self):
        """Generate all charts."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping chart generation.")
            return
        
        print("\n" + "=" * 120)
        print("GENERATING CHARTS AND FIGURES")
        print("=" * 120)
        
        self.chart_symmetric_performance()
        self.chart_asymmetric_performance()
        self.chart_fault_tolerance()
        self.chart_throughput_comparison()
        self.chart_mongodb_performance()
        self.chart_memory_safety()
        self.chart_performance_ranking()
        self.chart_algorithm_comparison_heatmap()
        
        print(f"\n✅ All charts saved to: {self.chart_dir}/")
        
    def chart_symmetric_performance(self):
        """Chart 1: Symmetric Encryption Performance Comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        algorithms = []
        enc_times = []
        dec_times = []
        
        if 'email' in self.results['symmetric']:
            for algo in self.results['symmetric']['email'].keys():
                if 1024 in self.results['symmetric']['email'][algo]:
                    algorithms.append(algo)
                    enc_times.append(self.results['symmetric']['email'][algo][1024]['encrypt_mean_us'])
                    dec_times.append(self.results['symmetric']['email'][algo][1024]['decrypt_mean_us'])
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        # Bar chart for encryption/decryption times
        ax1.bar(x - width/2, enc_times, width, label='Encryption', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, dec_times, width, label='Decryption', color='#A23B72', alpha=0.8)
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Time (μs)')
        ax1.set_title('Symmetric Encryption/Decryption Time (1KB Data)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bar chart for throughput
        throughput = []
        for algo in algorithms:
            if 'email' in self.results['symmetric'] and algo in self.results['symmetric']['email']:
                if 1024 in self.results['symmetric']['email'][algo]:
                    throughput.append(self.results['symmetric']['email'][algo][1024]['encrypt_throughput_mbps'])
        
        ax2.bar(algorithms, throughput, color='#F18F01', alpha=0.8)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Throughput (MB/s)')
        ax2.set_title('Symmetric Encryption Throughput (1KB Data)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(throughput):
            ax2.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/symmetric_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created: symmetric_performance.png")
    
    def chart_asymmetric_performance(self):
        """Chart 2: Asymmetric Key Wrapping Performance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        types = []
        wrap_times = []
        unwrap_times = []
        key_sizes = []
        
        for key_type, data in self.results['asymmetric'].items():
            types.append(data['key_type'] + '-' + str(data['key_size']))
            wrap_times.append(data['wrap_mean_us'])
            unwrap_times.append(data['unwrap_mean_us'])
            key_sizes.append(data['wrapped_key_size'])
        
        x = np.arange(len(types))
        width = 0.35
        
        # Bar chart for wrap/unwrap times
        ax1.bar(x - width/2, wrap_times, width, label='Wrap', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, unwrap_times, width, label='Unwrap', color='#C73E1D', alpha=0.8)
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Time (μs)')
        ax1.set_title('Asymmetric Key Wrapping Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(types, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log scale for better visualization
        ax1.set_yscale('log')
        
        # Bar chart for wrapped key sizes
        ax2.bar(types, key_sizes, color='#6A994E', alpha=0.8)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Wrapped Key Size (bytes)')
        ax2.set_title('Wrapped Key Size Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(key_sizes):
            ax2.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/asymmetric_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created: asymmetric_performance.png")
    
    def chart_fault_tolerance(self):
        """Chart 3: Fault Tolerance (Bit-Flip Resilience)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        algorithms = []
        success_rates = []
        damage_rates = []
        
        for algo, data in self.results['fault_tolerance'].items():
            algorithms.append(algo)
            success_rates.append(data['decryption_success_rate'] * 100)
            damage_rates.append(data['damage_ratio'] * 100)
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        # Bar chart for success rates
        colors = ['#6A994E' if rate > 80 else '#C73E1D' for rate in success_rates]
        ax1.bar(algorithms, success_rates, color=colors, alpha=0.8)
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Decryption Success Rate (%)')
        ax1.set_title('Bit-Flip Resilience: Decryption Success Rate')
        ax1.set_ylim(0, 105)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Acceptable threshold')
        ax1.legend()
        
        # Add value labels
        for i, v in enumerate(success_rates):
            ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Bar chart for damage rates
        colors = ['#6A994E' if d < 10 else '#F18F01' if d < 50 else '#C73E1D' for d in damage_rates]
        ax2.bar(algorithms, damage_rates, color=colors, alpha=0.8)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Average Data Damage (%)')
        ax2.set_title('Bit-Flip Resilience: Data Damage')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(damage_rates):
            ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/fault_tolerance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created: fault_tolerance.png")
    
    def chart_throughput_comparison(self):
        """Chart 4: Throughput Comparison Across Algorithms."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        algorithms = []
        encrypt_throughput = []
        decrypt_throughput = []
        
        if 'email' in self.results['symmetric']:
            for algo in self.results['symmetric']['email'].keys():
                if 1024 in self.results['symmetric']['email'][algo]:
                    algorithms.append(algo)
                    encrypt_throughput.append(self.results['symmetric']['email'][algo][1024]['encrypt_throughput_mbps'])
                    decrypt_throughput.append(self.results['symmetric']['email'][algo][1024]['decrypt_throughput_mbps'])
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        ax.bar(x - width/2, encrypt_throughput, width, label='Encryption', color='#2E86AB', alpha=0.8)
        ax.bar(x + width/2, decrypt_throughput, width, label='Decryption', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Throughput (MB/s)')
        ax.set_title('Symmetric Algorithm Throughput Comparison (1KB Data)')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (enc, dec) in enumerate(zip(encrypt_throughput, decrypt_throughput)):
            ax.text(i - width/2, enc + 5, f'{enc:.0f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, dec + 5, f'{dec:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/throughput_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created: throughput_comparison.png")
    
    def chart_mongodb_performance(self):
        """Chart 5: MongoDB Integration Performance."""
        if 'mongodb' not in self.results or 'standard' not in self.results['mongodb']:
            print("  ⚠ Skipping MongoDB chart (data not available)")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        modes = []
        times = []
        colors = []
        
        for mode, data in self.results['mongodb'].items():
            if mode != 'note':
                modes.append(mode.capitalize())
                times.append(data['insert_time_us'])
                colors.append('#2E86AB' if mode == 'standard' else '#6A994E')
        
        bars = ax.bar(modes, times, color=colors, alpha=0.8)
        ax.set_xlabel('Encryption Mode')
        ax.set_ylabel('Insert Time (μs)')
        ax.set_title('MongoDB Encrypted Document Insert Performance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                   f'{time_val:.0f} μs', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add speedup annotation if both modes exist
        if len(modes) == 2 and times[1] > 0:
            speedup = times[0] / times[1]
            ax.annotate(f'🚀 {speedup:.1f}x Faster',
                       xy=(1, times[1]), xytext=(1.3, times[1] + 500),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=12, color='green', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/mongodb_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created: mongodb_performance.png")
    
    def chart_memory_safety(self):
        """Chart 6: Memory Safety Analysis."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        algorithms = []
        memory_changes = []
        colors = []
        
        for algo, data in self.results['memory_safety'].items():
            algorithms.append(algo)
            memory_changes.append(data['memory_change_kb'])
            colors.append('#C73E1D' if data['memory_leak_suspected'] else '#6A994E')
        
        bars = ax.bar(algorithms, memory_changes, color=colors, alpha=0.8)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Memory Change (KB)')
        ax.set_title('Memory Safety: Memory Change After Encryption/Decryption')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, change in zip(bars, memory_changes):
            height = bar.get_height()
            label = f'{change:.2f} KB'
            if change > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       label, ha='center', va='bottom', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height - 0.5,
                       label, ha='center', va='top', fontsize=9)
        
        # Add legend
        legend_elements = [
            Rectangle((0,0),1,1, facecolor='#C73E1D', label='⚠️ Memory Leak Suspected'),
            Rectangle((0,0),1,1, facecolor='#6A994E', label='✅ Clean')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/memory_safety.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created: memory_safety.png")
    
    def chart_performance_ranking(self):
        """Chart 7: Performance Ranking Summary."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        categories = []
        winners = []
        values = []
        colors = []
        
        # Gather ranking data
        if 'email' in self.results['symmetric']:
            fastest_enc = min(self.results['symmetric']['email'].keys(),
                             key=lambda x: self.results['symmetric']['email'][x][1024]['encrypt_mean_us'])
            fastest_dec = min(self.results['symmetric']['email'].keys(),
                             key=lambda x: self.results['symmetric']['email'][x][1024]['decrypt_mean_us'])
            best_throughput = max(self.results['symmetric']['email'].keys(),
                                 key=lambda x: self.results['symmetric']['email'][x][1024]['encrypt_throughput_mbps'])
            
            categories.extend(['Fastest Encryption', 'Fastest Decryption', 'Best Throughput'])
            winners.extend([fastest_enc, fastest_dec, best_throughput])
            values.extend([
                f"{self.results['symmetric']['email'][fastest_enc][1024]['encrypt_mean_us']:.2f} μs",
                f"{self.results['symmetric']['email'][fastest_dec][1024]['decrypt_mean_us']:.2f} μs",
                f"{self.results['symmetric']['email'][best_throughput][1024]['encrypt_throughput_mbps']:.1f} MB/s"
            ])
        
        # Most resilient
        most_resilient = max(self.results['fault_tolerance'].keys(),
                           key=lambda x: self.results['fault_tolerance'][x]['decryption_success_rate'])
        categories.append('Most Resilient')
        winners.append(most_resilient)
        values.append(f"{self.results['fault_tolerance'][most_resilient]['decryption_success_rate']*100:.1f}%")
        
        # Key wrapping
        fastest_wrap = min(self.results['asymmetric'].keys(),
                          key=lambda x: self.results['asymmetric'][x]['wrap_mean_us'])
        fastest_unwrap = min(self.results['asymmetric'].keys(),
                            key=lambda x: self.results['asymmetric'][x]['unwrap_mean_us'])
        
        categories.extend(['Fastest Key Wrap', 'Fastest Key Unwrap'])
        winners.extend([fastest_wrap, fastest_unwrap])
        values.extend([
            f"{self.results['asymmetric'][fastest_wrap]['wrap_mean_us']:.2f} μs",
            f"{self.results['asymmetric'][fastest_unwrap]['unwrap_mean_us']:.2f} μs"
        ])
        
        # Create color map
        color_map = {
            'AES-GCM': '#2E86AB',
            'ChaCha20': '#A23B72',
            'AES-CBC': '#F18F01',
            'Blowfish': '#C73E1D',
            'RSA_2048': '#6A994E',
            'RSA_4096': '#BC4B51',
            'ECC_P256': '#5D576B',
            'ECC_P384': '#D4A5A5'
        }
        
        colors = [color_map.get(w, '#888888') for w in winners]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(categories))
        ax.barh(y_pos, [1] * len(categories), color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlabel('Performance Category')
        ax.set_title('Performance Rankings Summary')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        
        # Add winner and value labels
        for i, (winner, value) in enumerate(zip(winners, values)):
            ax.text(0.5, i, f'{winner}  ({value})', 
                   ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/performance_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created: performance_ranking.png")
    
    def chart_algorithm_comparison_heatmap(self):
        """Chart 8: Algorithm Comparison Heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if 'email' not in self.results['symmetric']:
            print("  ⚠ Skipping heatmap (data not available)")
            return
        
        # Prepare data for heatmap
        algorithms = list(self.results['symmetric']['email'].keys())
        metrics = ['Encrypt (μs)', 'Decrypt (μs)', 'Throughput (MB/s)', 'Memory (KB)']
        
        data = []
        for algo in algorithms:
            if 1024 in self.results['symmetric']['email'][algo]:
                row = [
                    self.results['symmetric']['email'][algo][1024]['encrypt_mean_us'],
                    self.results['symmetric']['email'][algo][1024]['decrypt_mean_us'],
                    self.results['symmetric']['email'][algo][1024]['encrypt_throughput_mbps'],
                    self.results['memory_safety'][algo]['memory_change_kb']
                ]
                data.append(row)
        
        # Normalize data for heatmap
        data_np = np.array(data)
        normalized_data = (data_np - data_np.min(axis=0)) / (data_np.max(axis=0) - data_np.min(axis=0))
        
        # Create heatmap
        im = ax.imshow(normalized_data, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(algorithms)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Relative Performance (Normalized)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(metrics)):
                if j < 3:  # For performance metrics, lower is better (green)
                    value = data[i][j]
                    text = f'{value:.2f}' if j < 2 else f'{value:.1f}'
                else:  # Memory - show as is
                    value = data[i][j]
                    text = f'{value:.2f}'
                ax.text(j, i, text, ha='center', va='center', 
                       color='white' if normalized_data[i][j] > 0.5 else 'black',
                       fontsize=9)
        
        ax.set_title('Algorithm Performance Comparison Heatmap\n(Green = Better Performance)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/algorithm_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created: algorithm_heatmap.png")
    
    def create_summary_figure(self):
        """Create a summary figure with multiple subplots."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Subplot 1: Symmetric Performance
        ax1 = fig.add_subplot(gs[0, 0])
        algorithms = []
        enc_times = []
        if 'email' in self.results['symmetric']:
            for algo in self.results['symmetric']['email'].keys():
                if 1024 in self.results['symmetric']['email'][algo]:
                    algorithms.append(algo)
                    enc_times.append(self.results['symmetric']['email'][algo][1024]['encrypt_mean_us'])
        ax1.bar(algorithms, enc_times, color='#2E86AB', alpha=0.8)
        ax1.set_title('Encryption Time')
        ax1.set_ylabel('μs')
        ax1.tick_params(axis='x', rotation=45)
        
        # Subplot 2: Asymmetric Performance
        ax2 = fig.add_subplot(gs[0, 1])
        types = []
        wrap_times = []
        for key_type, data in self.results['asymmetric'].items():
            types.append(data['key_type'] + '-' + str(data['key_size']))
            wrap_times.append(data['wrap_mean_us'])
        ax2.bar(types, wrap_times, color='#A23B72', alpha=0.8)
        ax2.set_title('Key Wrap Time')
        ax2.set_ylabel('μs')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yscale('log')
        
        # Subplot 3: Fault Tolerance
        ax3 = fig.add_subplot(gs[0, 2])
        algorithms = []
        success_rates = []
        for algo, data in self.results['fault_tolerance'].items():
            algorithms.append(algo)
            success_rates.append(data['decryption_success_rate'] * 100)
        colors = ['#6A994E' if rate > 80 else '#C73E1D' for rate in success_rates]
        ax3.bar(algorithms, success_rates, color=colors, alpha=0.8)
        ax3.set_title('Fault Tolerance')
        ax3.set_ylabel('Success Rate %')
        ax3.set_ylim(0, 105)
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=80, color='green', linestyle='--', alpha=0.5)
        
        # Subplot 4: Throughput
        ax4 = fig.add_subplot(gs[1, 0])
        throughput = []
        if 'email' in self.results['symmetric']:
            for algo in self.results['symmetric']['email'].keys():
                if 1024 in self.results['symmetric']['email'][algo]:
                    throughput.append(self.results['symmetric']['email'][algo][1024]['encrypt_throughput_mbps'])
        ax4.bar(algorithms, throughput, color='#F18F01', alpha=0.8)
        ax4.set_title('Throughput')
        ax4.set_ylabel('MB/s')
        ax4.tick_params(axis='x', rotation=45)
        
        # Subplot 5: MongoDB Performance
        ax5 = fig.add_subplot(gs[1, 1])
        if 'mongodb' in self.results and 'standard' in self.results['mongodb']:
            modes = []
            times = []
            for mode, data in self.results['mongodb'].items():
                if mode != 'note':
                    modes.append(mode.capitalize())
                    times.append(data['insert_time_us'])
            ax5.bar(modes, times, color=['#2E86AB', '#6A994E'], alpha=0.8)
            ax5.set_title('MongoDB Insert Time')
            ax5.set_ylabel('μs')
            
            # Add speedup annotation
            if len(times) == 2 and times[1] > 0:
                speedup = times[0] / times[1]
                ax5.annotate(f'{speedup:.1f}x faster', xy=(1, times[1]), 
                           xytext=(0.5, max(times) * 0.8),
                           arrowprops=dict(arrowstyle='->', color='green'),
                           fontsize=10, color='green')
        
        # Subplot 6: Memory Safety
        ax6 = fig.add_subplot(gs[1, 2])
        memory_changes = []
        for algo in self.results['memory_safety'].keys():
            memory_changes.append(self.results['memory_safety'][algo]['memory_change_kb'])
        colors = ['#C73E1D' if change > 2 else '#6A994E' for change in memory_changes]
        ax6.bar(self.results['memory_safety'].keys(), memory_changes, color=colors, alpha=0.8)
        ax6.set_title('Memory Change')
        ax6.set_ylabel('KB')
        ax6.tick_params(axis='x', rotation=45)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        fig.suptitle('Cryptographic Benchmark Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.chart_dir}/summary_figure.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created: summary_figure.png")

# ============================================================================
# MAIN BENCHMARK SUITE
# ============================================================================

class CryptoBenchmark:
    """Main benchmarking suite for cryptographic operations."""
    
    def __init__(self):
        self.providers = {
            'AES-GCM': AESGCMProvider(),
            'ChaCha20': ChaCha20Provider(),
            'AES-CBC': AESCBCProvider(),
            'Blowfish': BlowfishProvider(),
        }
        self.asymmetric_wrapper = AsymmetricKeyWrapper()
        
        if MONGO_AVAILABLE:
            self.db = EncryptedMongoDB()
        else:
            self.db = None
            
        self.data_gen = DataGenerator()
        self.results = {}
        
    def benchmark_symmetric_encryption(
        self,
        provider: CryptoProvider,
        data_size: int,
        iterations: int = 50
    ) -> Dict[str, Any]:
        """Benchmark symmetric encryption performance."""
        key = provider.generate_key()
        data = os.urandom(data_size)
        for _ in range(5):
            provider.encrypt(data, key)
        
        encrypt_times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            encrypted = provider.encrypt(data, key)
            end = time.perf_counter_ns()
            encrypt_times.append(end - start)
        
        decrypt_times = []
        encrypted_data = provider.encrypt(data, key)
        for _ in range(iterations):
            start = time.perf_counter_ns()
            decrypted = provider.decrypt(encrypted_data, key)
            end = time.perf_counter_ns()
            decrypt_times.append(end - start)
        
        encrypt_throughput = (data_size * iterations) / (sum(encrypt_times) / 1e9) / (1024 * 1024)
        decrypt_throughput = (data_size * iterations) / (sum(decrypt_times) / 1e9) / (1024 * 1024)
        
        return {
            'algorithm': provider.get_name(),
            'key_size': provider.get_key_size(),
            'data_size': data_size,
            'data_size_mb': data_size / (1024 * 1024),
            'encrypt_mean_us': np.mean(encrypt_times) / 1000,
            'encrypt_std_us': np.std(encrypt_times) / 1000,
            'encrypt_p50_us': np.percentile(encrypt_times, 50) / 1000,
            'encrypt_p95_us': np.percentile(encrypt_times, 95) / 1000,
            'decrypt_mean_us': np.mean(decrypt_times) / 1000,
            'decrypt_std_us': np.std(decrypt_times) / 1000,
            'decrypt_p50_us': np.percentile(decrypt_times, 50) / 1000,
            'decrypt_p95_us': np.percentile(decrypt_times, 95) / 1000,
            'encrypt_throughput_mbps': encrypt_throughput,
            'decrypt_throughput_mbps': decrypt_throughput,
            'iterations': iterations
        }
    
    def benchmark_asymmetric_wrapping(
        self,
        key_type: str,
        key_size: int = 2048,
        iterations: int = 20
    ) -> Dict[str, Any]:
        """Benchmark asymmetric key wrapping performance."""
        dek = os.urandom(32)
        
        if key_type == 'rsa':
            private_key, public_key = self.asymmetric_wrapper.generate_rsa_keypair(key_size)
            
            for _ in range(3):
                self.asymmetric_wrapper.wrap_key_rsa(dek, public_key)
            
            wrap_times = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                wrapped = self.asymmetric_wrapper.wrap_key_rsa(dek, public_key)
                end = time.perf_counter_ns()
                wrap_times.append(end - start)
            
            unwrap_times = []
            wrapped_sample = self.asymmetric_wrapper.wrap_key_rsa(dek, public_key)
            for _ in range(iterations):
                start = time.perf_counter_ns()
                unwrapped = self.asymmetric_wrapper.unwrap_key_rsa(wrapped_sample, private_key)
                end = time.perf_counter_ns()
                unwrap_times.append(end - start)
            
            return {
                'key_type': 'RSA',
                'key_size': key_size,
                'security_bits': key_size // 2,
                'wrap_mean_us': np.mean(wrap_times) / 1000,
                'wrap_std_us': np.std(wrap_times) / 1000,
                'unwrap_mean_us': np.mean(unwrap_times) / 1000,
                'unwrap_std_us': np.std(unwrap_times) / 1000,
                'wrapped_key_size': len(wrapped_sample),
                'iterations': iterations
            }
        
        else:  # ECC
            if key_size == 256:
                curve = ec.SECP256R1()
                security_bits = 128
            else:
                curve = ec.SECP384R1()
                security_bits = 192
            
            private_key, public_key = self.asymmetric_wrapper.generate_ecc_keypair(curve)
            
            wrap_times = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                wrapped = self.asymmetric_wrapper.wrap_key_ecc(dek, public_key)
                end = time.perf_counter_ns()
                wrap_times.append(end - start)
            
            unwrap_times = []
            wrapped_sample = self.asymmetric_wrapper.wrap_key_ecc(dek, public_key)
            for _ in range(iterations):
                start = time.perf_counter_ns()
                unwrapped = self.asymmetric_wrapper.unwrap_key_ecc(wrapped_sample, private_key)
                end = time.perf_counter_ns()
                unwrap_times.append(end - start)
            
            return {
                'key_type': 'ECC',
                'key_size': key_size,
                'security_bits': security_bits,
                'curve': 'P-256' if key_size == 256 else 'P-384',
                'wrap_mean_us': np.mean(wrap_times) / 1000,
                'wrap_std_us': np.std(wrap_times) / 1000,
                'unwrap_mean_us': np.mean(unwrap_times) / 1000,
                'unwrap_std_us': np.std(unwrap_times) / 1000,
                'wrapped_key_size': len(wrapped_sample),
                'iterations': iterations
            }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all algorithms and data types."""
        results = {
            'symmetric': {},
            'asymmetric': {},
            'fault_tolerance': {},
            'memory_safety': {},
            'mongodb': {},
        }
        
        print("\n  Running symmetric encryption benchmarks...")
        for data_type in config.DATA_TYPES:
            results['symmetric'][data_type] = {}
            for provider_name, provider in self.providers.items():
                results['symmetric'][data_type][provider_name] = {}
                for size in config.PAYLOAD_SIZES[:3]:
                    data = DataGenerator.generate_payload_by_type(data_type, size)
                    key = provider.generate_key()
                    results['symmetric'][data_type][provider_name][size] = self.benchmark_symmetric_encryption(
                        provider, size, config.ITERATIONS
                    )
        
        print("  Running asymmetric key wrapping benchmarks...")
        results['asymmetric']['RSA_2048'] = self.benchmark_asymmetric_wrapping('rsa', 2048, config.ASYMMETRIC_ITERATIONS)
        results['asymmetric']['RSA_4096'] = self.benchmark_asymmetric_wrapping('rsa', 4096, config.ASYMMETRIC_ITERATIONS)
        results['asymmetric']['ECC_P256'] = self.benchmark_asymmetric_wrapping('ecc', 256, config.ASYMMETRIC_ITERATIONS)
        results['asymmetric']['ECC_P384'] = self.benchmark_asymmetric_wrapping('ecc', 384, config.ASYMMETRIC_ITERATIONS)
        
        print("  Running fault tolerance tests...")
        for provider_name, provider in self.providers.items():
            test_data = DataGenerator.generate_binary_file(1024)
            key = provider.generate_key()
            results['fault_tolerance'][provider_name] = FaultInjector.test_bit_flip_resilience(
                provider, test_data, key, num_flips=10
            )
        
        print("  Running memory safety tests...")
        for provider_name, provider in self.providers.items():
            test_data = DataGenerator.generate_binary_file(1024)
            key = provider.generate_key()
            results['memory_safety'][provider_name] = MemorySafetyTester.test_memory_cleanup(
                provider, test_data, key
            )
        
        if MONGO_AVAILABLE and self.db:
            print("  Running MongoDB integration tests...")
            results['mongodb']['standard'] = self.test_mongodb_integration(use_hybrid=False)
            results['mongodb']['hybrid'] = self.test_mongodb_integration(use_hybrid=True)
        else:
            results['mongodb']['note'] = "MongoDB not available - integration tests skipped"
        
        return results
    
    def test_mongodb_integration(self, use_hybrid: bool = False) -> Dict[str, Any]:
        """Test MongoDB integration with encrypted documents."""
        if not MONGO_AVAILABLE or not self.db:
            return {'error': 'MongoDB not available'}
        
        collection = self.db.get_collection('test_encrypted')
        
        doc = {
            'email': self.data_gen.generate_email(),
            'profile': self.data_gen.generate_json_document(512),
            'uuid': self.data_gen.generate_uuid(),
        }
        
        encryption_keys = {
            'email': os.urandom(32),
            'profile': os.urandom(32),
            'uuid': os.urandom(32),
        }
        
        algorithm_mapping = {
            'email': self.providers['AES-GCM'],
            'profile': self.providers['ChaCha20'],
            'uuid': self.providers['AES-CBC'],
        }
        
        hybrid_public_key = None
        if use_hybrid:
            private_key, hybrid_public_key = self.asymmetric_wrapper.generate_rsa_keypair(2048)
        
        start = time.perf_counter_ns()
        doc_id = self.db.insert_encrypted_document(
            collection,
            doc,
            encryption_keys,
            algorithm_mapping,
            hybrid_public_key=hybrid_public_key,
            use_hybrid=use_hybrid
        )
        insert_time = (time.perf_counter_ns() - start) / 1000
        
        return {
            'inserted_id': doc_id,
            'insert_time_us': insert_time,
            'use_hybrid': use_hybrid
        }
    
    def print_comprehensive_tables(self, results: Dict[str, Any]):
        """Print ALL results in comprehensive table format."""
        print("\n" + "=" * 120)
        print("COMPREHENSIVE BENCHMARK RESULTS - ALL TABLES")
        print("=" * 120)
        
        # ========================================================================
        # TABLE 1: SYMMETRIC ENCRYPTION - ALL DATA TYPES AND SIZES
        # ========================================================================
        print("\n" + "=" * 120)
        print("TABLE 1: SYMMETRIC ENCRYPTION PERFORMANCE - ALL DATA TYPES")
        print("=" * 120)
        
        for data_type in config.DATA_TYPES:
            if data_type not in results['symmetric']:
                continue
                
            print(f"\nData Type: {data_type.upper()}")
            print("-" * 120)
            
            table_data = []
            for algo in self.providers.keys():
                if algo in results['symmetric'][data_type]:
                    for size in sorted(results['symmetric'][data_type][algo].keys()):
                        r = results['symmetric'][data_type][algo][size]
                        table_data.append([
                            algo,
                            r['key_size'],
                            size,
                            f"{r['encrypt_mean_us']:.2f}",
                            f"{r['decrypt_mean_us']:.2f}",
                            f"{r['encrypt_throughput_mbps']:.1f}",
                            f"{r['decrypt_throughput_mbps']:.1f}",
                            f"{r['encrypt_p95_us']:.2f}",
                            f"{r['decrypt_p95_us']:.2f}",
                        ])
            
            if TABULATE_AVAILABLE:
                headers = ["Algorithm", "Key(bits)", "Size(B)", "Enc(μs)", "Dec(μs)", 
                          "Enc MB/s", "Dec MB/s", "Enc P95", "Dec P95"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
            else:
                print("Algorithm | Key(bits) | Size(B) | Enc(μs) | Dec(μs) | Enc MB/s | Dec MB/s | Enc P95 | Dec P95")
                print("-" * 100)
                for row in table_data:
                    print(f"{row[0]:<10} | {row[1]:<8} | {row[2]:<6} | {row[3]:<7} | {row[4]:<7} | {row[5]:<8} | {row[6]:<8} | {row[7]:<7} | {row[8]:<7}")
        
        # ========================================================================
        # TABLE 2: SYMMETRIC ENCRYPTION - SUMMARY (1KB Data)
        # ========================================================================
        print("\n" + "=" * 120)
        print("TABLE 2: SYMMETRIC ENCRYPTION SUMMARY (1KB Data)")
        print("=" * 120)
        
        table_data = []
        for data_type in ['email', 'json_document']:
            if data_type not in results['symmetric']:
                continue
            for algo in self.providers.keys():
                if 1024 in results['symmetric'][data_type][algo]:
                    r = results['symmetric'][data_type][algo][1024]
                    table_data.append([
                        data_type,
                        algo,
                        r['key_size'],
                        f"{r['encrypt_mean_us']:.2f}",
                        f"{r['decrypt_mean_us']:.2f}",
                        f"{r['encrypt_throughput_mbps']:.1f}",
                        f"{r['decrypt_throughput_mbps']:.1f}",
                        f"{r['encrypt_p95_us']:.2f}",
                        f"{r['decrypt_p95_us']:.2f}",
                    ])
        
        if TABULATE_AVAILABLE:
            headers = ["Data Type", "Algorithm", "Key(bits)", "Enc(μs)", "Dec(μs)", 
                      "Enc MB/s", "Dec MB/s", "Enc P95", "Dec P95"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            print("Data Type | Algorithm | Key(bits) | Enc(μs) | Dec(μs) | Enc MB/s | Dec MB/s | Enc P95 | Dec P95")
            print("-" * 100)
            for row in table_data:
                print(f"{row[0]:<10} | {row[1]:<12} | {row[2]:<8} | {row[3]:<7} | {row[4]:<7} | {row[5]:<8} | {row[6]:<8} | {row[7]:<7} | {row[8]:<7}")
        
        # ========================================================================
        # TABLE 3: ASYMMETRIC KEY WRAPPING PERFORMANCE
        # ========================================================================
        print("\n" + "=" * 120)
        print("TABLE 3: ASYMMETRIC KEY WRAPPING PERFORMANCE")
        print("=" * 120)
        
        asym_data = []
        for key_type, r in results['asymmetric'].items():
            asym_data.append([
                r['key_type'],
                f"{r['key_size']}",
                f"{r['security_bits']}",
                f"{r['wrap_mean_us']:.2f}",
                f"{r['wrap_std_us']:.2f}",
                f"{r['unwrap_mean_us']:.2f}",
                f"{r['unwrap_std_us']:.2f}",
                f"{r['wrapped_key_size']}",
                f"{r['unwrap_mean_us'] / r['wrap_mean_us']:.1f}x" if r['wrap_mean_us'] > 0 else "N/A"
            ])
        
        if TABULATE_AVAILABLE:
            headers = ["Type", "Key Size", "Security(bits)", "Wrap(μs)", "Wrap Std", 
                      "Unwrap(μs)", "Unwrap Std", "Wrapped(B)", "Ratio"]
            print(tabulate(asym_data, headers=headers, tablefmt="grid"))
        else:
            print("Type | Key Size | Security(bits) | Wrap(μs) | Wrap Std | Unwrap(μs) | Unwrap Std | Wrapped(B) | Ratio")
            print("-" * 110)
            for row in asym_data:
                print(f"{row[0]:<6} | {row[1]:<8} | {row[2]:<13} | {row[3]:<8} | {row[4]:<8} | {row[5]:<10} | {row[6]:<10} | {row[7]:<10} | {row[8]}")
        
        # ========================================================================
        # TABLE 4: FAULT TOLERANCE (BIT-FLIP RESILIENCE)
        # ========================================================================
        print("\n" + "=" * 120)
        print("TABLE 4: FAULT TOLERANCE - BIT-FLIP RESILIENCE")
        print("=" * 120)
        
        fault_data = []
        for algo, r in results['fault_tolerance'].items():
            success_rate = r['decryption_success_rate'] * 100
            damage_pct = r['damage_ratio'] * 100
            status = "✅ PASS" if success_rate > 80 else "❌ FAIL"
            fault_data.append([
                algo,
                f"{success_rate:.1f}%",
                f"{damage_pct:.1f}%",
                f"{r['avg_damaged_bytes']:.1f}",
                f"{r['ciphertext_size']}",
                f"{r['plaintext_size']}",
                status,
                "Authenticated" if algo in ['AES-GCM', 'ChaCha20'] else "Non-Authenticated"
            ])
        
        if TABULATE_AVAILABLE:
            headers = ["Algorithm", "Decrypt Success", "Avg Damage", "Avg Damaged Bytes", 
                      "Ciphertext Size", "Plaintext Size", "Status", "Mode"]
            print(tabulate(fault_data, headers=headers, tablefmt="grid"))
        else:
            print("Algorithm | Decrypt Success | Avg Damage | Avg Damaged Bytes | Ciphertext Size | Plaintext Size | Status | Mode")
            print("-" * 120)
            for row in fault_data:
                print(f"{row[0]:<12} | {row[1]:<15} | {row[2]:<11} | {row[3]:<17} | {row[4]:<14} | {row[5]:<13} | {row[6]:<8} | {row[7]}")
        
        # ========================================================================
        # TABLE 5: MEMORY SAFETY
        # ========================================================================
        print("\n" + "=" * 120)
        print("TABLE 5: MEMORY SAFETY ANALYSIS")
        print("=" * 120)
        
        mem_data = []
        for algo, r in results['memory_safety'].items():
            mem_change = r['memory_change_kb']
            status = "⚠️ LEAK" if r['memory_leak_suspected'] else "✅ CLEAN"
            mem_data.append([
                algo,
                f"{r['initial_memory']}",
                f"{r['memory_after_encrypt']}",
                f"{r['memory_final']}",
                f"{mem_change:.2f}",
                f"{mem_change / r['plaintext_size'] * 1024:.2f}%",
                status,
                r['plaintext_size']
            ])
        
        if TABULATE_AVAILABLE:
            headers = ["Algorithm", "Initial(B)", "After Encrypt(B)", "Final(B)", 
                      "Change(KB)", "Change %", "Status", "Plaintext Size"]
            print(tabulate(mem_data, headers=headers, tablefmt="grid"))
        else:
            print("Algorithm | Initial(B) | After Encrypt(B) | Final(B) | Change(KB) | Change % | Status | Plaintext Size")
            print("-" * 120)
            for row in mem_data:
                print(f"{row[0]:<12} | {row[1]:<10} | {row[2]:<15} | {row[3]:<8} | {row[4]:<10} | {row[5]:<9} | {row[6]:<8} | {row[7]}")
        
        # ========================================================================
        # TABLE 6: MONGODB INTEGRATION PERFORMANCE
        # ========================================================================
        print("\n" + "=" * 120)
        print("TABLE 6: MONGODB INTEGRATION PERFORMANCE")
        print("=" * 120)
        
        if 'mongodb' in results and 'note' not in results['mongodb']:
            mongo_data = []
            for mode, r in results['mongodb'].items():
                if mode != 'note':
                    speedup = "N/A"
                    if mode == 'hybrid' and 'standard' in results['mongodb']:
                        std_time = results['mongodb']['standard'].get('insert_time_us', 0)
                        if std_time > 0:
                            speedup = f"{std_time / r['insert_time_us']:.1f}x"
                    
                    mongo_data.append([
                        mode.capitalize(),
                        r['inserted_id'][:20] + "...",
                        f"{r['insert_time_us']:.2f}",
                        "Hybrid" if r.get('use_hybrid', False) else "Standard",
                        speedup,
                        "✅" if r.get('insert_time_us', 0) < 3000 else "⚠️"
                    ])
            
            if TABULATE_AVAILABLE:
                headers = ["Mode", "Document ID", "Insert Time(μs)", "Encryption Type", "Speedup", "Status"]
                print(tabulate(mongo_data, headers=headers, tablefmt="grid"))
            else:
                print("Mode | Document ID | Insert Time(μs) | Encryption Type | Speedup | Status")
                print("-" * 80)
                for row in mongo_data:
                    print(f"{row[0]:<8} | {row[1]:<22} | {row[2]:<14} | {row[3]:<15} | {row[4]:<7} | {row[5]}")
        else:
            print("MongoDB integration tests skipped")
        
        # ========================================================================
        # TABLE 7: PERFORMANCE RANKING
        # ========================================================================
        print("\n" + "=" * 120)
        print("TABLE 7: PERFORMANCE RANKING SUMMARY")
        print("=" * 120)
        
        # Find fastest for each metric
        ranking_data = []
        
        # Encryption speed ranking
        enc_speeds = []
        for algo in self.providers.keys():
            if 'email' in results['symmetric'] and 1024 in results['symmetric']['email'][algo]:
                enc_speeds.append((algo, results['symmetric']['email'][algo][1024]['encrypt_mean_us']))
        enc_speeds.sort(key=lambda x: x[1])
        
        for i, (algo, speed) in enumerate(enc_speeds, 1):
            ranking_data.append([
                f"Encrypt Speed",
                algo,
                f"{speed:.2f} μs",
                f"#{i}",
                "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
            ])
        
        # Decryption speed ranking
        dec_speeds = []
        for algo in self.providers.keys():
            if 'email' in results['symmetric'] and 1024 in results['symmetric']['email'][algo]:
                dec_speeds.append((algo, results['symmetric']['email'][algo][1024]['decrypt_mean_us']))
        dec_speeds.sort(key=lambda x: x[1])
        
        for i, (algo, speed) in enumerate(dec_speeds, 1):
            ranking_data.append([
                f"Decrypt Speed",
                algo,
                f"{speed:.2f} μs",
                f"#{i}",
                "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
            ])
        
        # Throughput ranking
        throughput_speeds = []
        for algo in self.providers.keys():
            if 'email' in results['symmetric'] and 1024 in results['symmetric']['email'][algo]:
                throughput_speeds.append((algo, results['symmetric']['email'][algo][1024]['encrypt_throughput_mbps']))
        throughput_speeds.sort(key=lambda x: x[1], reverse=True)
        
        for i, (algo, speed) in enumerate(throughput_speeds, 1):
            ranking_data.append([
                f"Throughput",
                algo,
                f"{speed:.1f} MB/s",
                f"#{i}",
                "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
            ])
        
        # Fault tolerance ranking
        fault_ranking = []
        for algo, r in results['fault_tolerance'].items():
            fault_ranking.append((algo, r['decryption_success_rate']))
        fault_ranking.sort(key=lambda x: x[1], reverse=True)
        
        for i, (algo, rate) in enumerate(fault_ranking, 1):
            ranking_data.append([
                f"Fault Tolerance",
                algo,
                f"{rate*100:.1f}%",
                f"#{i}",
                "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
            ])
        
        if TABULATE_AVAILABLE:
            headers = ["Category", "Algorithm", "Value", "Rank", "Medal"]
            print(tabulate(ranking_data, headers=headers, tablefmt="grid"))
        else:
            print("Category | Algorithm | Value | Rank | Medal")
            print("-" * 60)
            for row in ranking_data:
                print(f"{row[0]:<15} | {row[1]:<12} | {row[2]:<10} | {row[3]:<4} | {row[4]}")
        
        # ========================================================================
        # TABLE 8: RECOMMENDATION MATRIX
        # ========================================================================
        print("\n" + "=" * 120)
        print("TABLE 8: RECOMMENDATION MATRIX")
        print("=" * 120)
        
        recommendations = [
            ["High Security + Integrity", "AES-256-GCM or ChaCha20-Poly1305", 
             "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "✅ Best for most apps"],
            ["Maximum Speed (x86)", "AES-256-GCM (with AES-NI)", 
             "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "✅ Hardware accelerated"],
            ["Maximum Speed (ARM/IoT)", "ChaCha20-Poly1305", 
             "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "✅ No hardware needed"],
            ["Fault Tolerant (Archival)", "AES-256-CBC", 
             "⭐⭐⭐", "⭐⭐⭐", "⚠️ Less secure but recoverable"],
            ["Key Management (Write-heavy)", "RSA-2048", 
             "⭐⭐⭐⭐", "⭐⭐⭐", "✅ Fast wrap"],
            ["Key Management (Read-heavy)", "ECC P-256", 
             "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "✅ Fast unwrap"],
            ["MongoDB Searchable", "AES-GCM + Blind Index", 
             "⭐⭐⭐⭐", "⭐⭐⭐", "✅ Searchable encryption"],
            ["Legacy Systems", "Blowfish", 
             "⭐⭐", "⭐⭐", "⚠️ Deprecated, avoid"],
        ]
        
        if TABULATE_AVAILABLE:
            headers = ["Use Case", "Recommended Algorithm", "Security", "Performance", "Notes"]
            print(tabulate(recommendations, headers=headers, tablefmt="grid"))
        else:
            print("Use Case | Recommended Algorithm | Security | Performance | Notes")
            print("-" * 100)
            for row in recommendations:
                print(f"{row[0]:<30} | {row[1]:<35} | {row[2]:<8} | {row[3]:<10} | {row[4]}")
        
        # ========================================================================
        # TABLE 9: KEY INSIGHTS
        # ========================================================================
        print("\n" + "=" * 120)
        print("TABLE 9: KEY INSIGHTS & STATISTICAL SUMMARY")
        print("=" * 120)
        
        insights = []
        
        # Fastest algorithm
        if 'email' in results['symmetric']:
            fastest_enc = min(self.providers.keys(), 
                             key=lambda x: results['symmetric']['email'][x][1024]['encrypt_mean_us'])
            fastest_dec = min(self.providers.keys(), 
                             key=lambda x: results['symmetric']['email'][x][1024]['decrypt_mean_us'])
            best_throughput = max(self.providers.keys(),
                                 key=lambda x: results['symmetric']['email'][x][1024]['encrypt_throughput_mbps'])
            
            insights.append(["Fastest Encryption", fastest_enc, 
                             f"{results['symmetric']['email'][fastest_enc][1024]['encrypt_mean_us']:.2f} μs"])
            insights.append(["Fastest Decryption", fastest_dec, 
                             f"{results['symmetric']['email'][fastest_dec][1024]['decrypt_mean_us']:.2f} μs"])
            insights.append(["Best Throughput", best_throughput, 
                             f"{results['symmetric']['email'][best_throughput][1024]['encrypt_throughput_mbps']:.1f} MB/s"])
        
        # Most resilient
        most_resilient = max(results['fault_tolerance'].keys(),
                           key=lambda x: results['fault_tolerance'][x]['decryption_success_rate'])
        insights.append(["Most Resilient", most_resilient, 
                         f"{results['fault_tolerance'][most_resilient]['decryption_success_rate']*100:.1f}% success"])
        
        # Fastest key wrap
        fastest_wrap = min(results['asymmetric'].keys(),
                          key=lambda x: results['asymmetric'][x]['wrap_mean_us'])
        insights.append(["Fastest Key Wrap", fastest_wrap, 
                         f"{results['asymmetric'][fastest_wrap]['wrap_mean_us']:.2f} μs"])
        
        # Fastest key unwrap
        fastest_unwrap = min(results['asymmetric'].keys(),
                            key=lambda x: results['asymmetric'][x]['unwrap_mean_us'])
        insights.append(["Fastest Key Unwrap", fastest_unwrap, 
                         f"{results['asymmetric'][fastest_unwrap]['unwrap_mean_us']:.2f} μs"])
        
        # Smallest wrapped key
        smallest_key = min(results['asymmetric'].keys(),
                          key=lambda x: results['asymmetric'][x]['wrapped_key_size'])
        insights.append(["Smallest Wrapped Key", smallest_key, 
                         f"{results['asymmetric'][smallest_key]['wrapped_key_size']} bytes"])
        
        if 'mongodb' in results and 'standard' in results['mongodb'] and 'hybrid' in results['mongodb']:
            speedup = results['mongodb']['standard']['insert_time_us'] / results['mongodb']['hybrid']['insert_time_us']
            insights.append(["MongoDB Hybrid Speedup", f"{speedup:.1f}x", 
                             f"{results['mongodb']['standard']['insert_time_us']:.0f}→{results['mongodb']['hybrid']['insert_time_us']:.0f} μs"])
        
        if TABULATE_AVAILABLE:
            headers = ["Metric", "Winner", "Value"]
            print(tabulate(insights, headers=headers, tablefmt="grid"))
        else:
            print("Metric | Winner | Value")
            print("-" * 50)
            for row in insights:
                print(f"{row[0]:<25} | {row[1]:<20} | {row[2]}")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main function to run the benchmark suite."""
    print("=" * 120)
    print("CRYPTOGRAPHIC BENCHMARK SUITE FOR MONGODB")
    print("Research Paper Implementation - Enhanced Version with Charts")
    print("=" * 120)
    
    crypto_version = __import__('cryptography').__version__
    
    print(f"\nSystem Information:")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  Cryptography Version: {crypto_version}")
    
    try:
        import pymongo
        print(f"  PyMongo Version: {pymongo.__version__}")
    except:
        print(f"  PyMongo Version: Not installed")
    
    print(f"  Data Sizes: {config.PAYLOAD_SIZES[:3]} bytes")
    print(f"  Iterations: {config.ITERATIONS}")
    print(f"  Data Types: {', '.join(config.DATA_TYPES)}")
    
    if MATPLOTLIB_AVAILABLE:
        print(f"  Matplotlib Version: {plt.matplotlib.__version__}")
    else:
        print(f"  Matplotlib: Not installed (charts will be skipped)")
    
    # Run demos
    print("\n" + "=" * 120)
    print("DEMONSTRATIONS")
    print("=" * 120)
    
    demo_individual_symmetric_encryption()
    demo_individual_asymmetric_encryption()
    demo_hybrid_encryption()
    
    # Initialize benchmark
    benchmark = CryptoBenchmark()
    
    # Run comprehensive benchmarks
    print("\n" + "=" * 120)
    print("COMPREHENSIVE BENCHMARKS")
    print("=" * 120)
    
    print("\n[1] Running Comprehensive Benchmarks...")
    start_time = time.time()
    results = benchmark.run_comprehensive_benchmark()
    elapsed = time.time() - start_time
    
    print(f"\nBenchmark completed in {elapsed:.2f} seconds")
    
    # Print comprehensive tables
    benchmark.print_comprehensive_tables(results)
    
    # Generate charts
    if MATPLOTLIB_AVAILABLE:
        chart_gen = ChartGenerator(results)
        chart_gen.generate_all_charts()
        chart_gen.create_summary_figure()
    
    # Save results to JSON
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    serializable_results = {}
    for key, value in results.items():
        if key == 'mongodb':
            serializable_results[key] = value
        elif key == 'fault_tolerance':
            serializable_results[key] = {
                k: {k2: v2 for k2, v2 in v.items() if k2 != 'sample_results'}
                for k, v in value.items()
            }
        elif key == 'symmetric':
            serializable_results[key] = {
                dt: {
                    algo: {
                        str(size): {
                            k: v for k, v in results['symmetric'][dt][algo][size].items()
                            if k != 'iterations'
                        }
                        for size in results['symmetric'][dt][algo]
                    }
                    for algo in results['symmetric'][dt]
                }
                for dt in results['symmetric']
            }
        else:
            serializable_results[key] = value
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, default=str, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 120)

if __name__ == "__main__":
    main()
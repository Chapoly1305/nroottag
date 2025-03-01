from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec

# COPY THE PRIVATE KEY HERE
priv_hex = "eff87411a86feb4cd51928136014d0ba4be39d0e1fe644e4e5ea5dac"

private_key = ec.derive_private_key(
    int(priv_hex, 16),
    ec.SECP224R1(),
    default_backend()
)
private_key_bytes = private_key.private_numbers().private_value.to_bytes(28, byteorder='big')
public_key = private_key.public_key()
public_key_bytes = public_key.public_numbers().x.to_bytes(28, byteorder='big')
print(f"PublicKeyHex: {public_key_bytes.hex()}")

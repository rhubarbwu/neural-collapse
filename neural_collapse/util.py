from hashlib import sha256

hashify = lambda O: sha256(O.cpu().numpy().tobytes()).hexdigest()
resolve = lambda A, B: B if not A or A == B else None

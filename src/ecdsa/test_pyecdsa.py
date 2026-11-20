from __future__ import with_statement, division, print_function

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import os
import shutil
import subprocess
import pytest
import sys
from binascii import hexlify, unhexlify
import hashlib
from functools import partial

from hypothesis import given, settings
import hypothesis.strategies as st

from six import binary_type
from .keys import SigningKey, VerifyingKey
from .keys import BadSignatureError, MalformedPointError, BadDigestError
from . import util
from .util import (
    sigencode_der,
    sigencode_strings,
    sigencode_strings_canonize,
    sigencode_string_canonize,
    sigencode_der_canonize,
)
from .util import sigdecode_der, sigdecode_strings, sigdecode_string
from .util import number_to_string, encoded_oid_ecPublicKey, MalformedSignature
from .curves import Curve, UnknownCurveError
from .curves import (
    SECP112r1,
    SECP112r2,
    SECP128r1,
    SECP160r1,
    NIST192p,
    NIST224p,
    NIST256p,
    NIST384p,
    NIST521p,
    SECP256k1,
    BRAINPOOLP160r1,
    BRAINPOOLP192r1,
    BRAINPOOLP224r1,
    BRAINPOOLP256r1,
    BRAINPOOLP320r1,
    BRAINPOOLP384r1,
    BRAINPOOLP512r1,
    BRAINPOOLP160t1,
    BRAINPOOLP192t1,
    BRAINPOOLP224t1,
    BRAINPOOLP256t1,
    BRAINPOOLP320t1,
    BRAINPOOLP384t1,
    BRAINPOOLP512t1,
    Ed25519,
    Ed448,
    curves,
)
from .ecdsa import (
    curve_brainpoolp224r1,
    curve_brainpoolp256r1,
    curve_brainpoolp384r1,
    curve_brainpoolp512r1,
)
from .ellipticcurve import Point
from . import der
from . import rfc6979
from . import ecdsa


class SubprocessError(Exception):
    pass


HYP_SETTINGS = {}


if "--fast" in sys.argv:  # pragma: no cover
    HYP_SETTINGS["max_examples"] = 2


def run_openssl(cmd):
    OPENSSL = "openssl"
    p = subprocess.Popen(
        [OPENSSL] + cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, ignored = p.communicate()
    if p.returncode != 0:
        raise SubprocessError(
            "cmd '%s %s' failed: rc=%s, stdout/err was %s"
            % (OPENSSL, cmd, p.returncode, stdout)
        )
    return stdout.decode()


class ECDSA(unittest.TestCase):
    def test_basic(self):
        priv = SigningKey.generate()
        pub = priv.get_verifying_key()

        data = b"blahblah"
        sig = priv.sign(data)

        self.assertTrue(pub.verify(sig, data))
        self.assertRaises(BadSignatureError, pub.verify, sig, data + b"bad")

        pub2 = VerifyingKey.from_string(pub.to_string())
        self.assertTrue(pub2.verify(sig, data))

    def test_deterministic(self):
        data = b"blahblah"
        secexp = int("9d0219792467d7d37b4d43298a7d0c05", 16)

        priv = SigningKey.from_secret_exponent(
            secexp, SECP256k1, hashlib.sha256
        )
        pub = priv.get_verifying_key()

        k = rfc6979.generate_k(
            SECP256k1.generator.order(),
            secexp,
            hashlib.sha256,
            hashlib.sha256(data).digest(),
        )

        sig1 = priv.sign(data, k=k)
        self.assertTrue(pub.verify(sig1, data))

        sig2 = priv.sign(data, k=k)
        self.assertTrue(pub.verify(sig2, data))

        sig3 = priv.sign_deterministic(data, hashlib.sha256)
        self.assertTrue(pub.verify(sig3, data))

        self.assertEqual(sig1, sig2)
        self.assertEqual(sig1, sig3)

    def test_bad_usage(self):
        # sk=SigningKey() is wrong
        self.assertRaises(TypeError, SigningKey)
        self.assertRaises(TypeError, VerifyingKey)

    def test_lengths_default(self):
        default = NIST192p
        priv = SigningKey.generate()
        pub = priv.get_verifying_key()
        self.assertEqual(len(pub.to_string()), default.verifying_key_length)
        sig = priv.sign(b"data")
        self.assertEqual(len(sig), default.signature_length)

    def test_serialize(self):
        seed = b"secret"
        curve = NIST192p
        secexp1 = util.randrange_from_seed__trytryagain(seed, curve.order)
        secexp2 = util.randrange_from_seed__trytryagain(seed, curve.order)
        self.assertEqual(secexp1, secexp2)
        priv1 = SigningKey.from_secret_exponent(secexp1, curve)
        priv2 = SigningKey.from_secret_exponent(secexp2, curve)
        self.assertEqual(
            hexlify(priv1.to_string()), hexlify(priv2.to_string())
        )
        self.assertEqual(priv1.to_pem(), priv2.to_pem())
        pub1 = priv1.get_verifying_key()
        pub2 = priv2.get_verifying_key()
        data = b"data"
        sig1 = priv1.sign(data)
        sig2 = priv2.sign(data)
        self.assertTrue(pub1.verify(sig1, data))
        self.assertTrue(pub2.verify(sig1, data))
        self.assertTrue(pub1.verify(sig2, data))
        self.assertTrue(pub2.verify(sig2, data))
        self.assertEqual(hexlify(pub1.to_string()), hexlify(pub2.to_string()))

    def test_nonrandom(self):
        s = b"all the entropy in the entire world, compressed into one line"

        def not_much_entropy(numbytes):
            return s[:numbytes]

        # we control the entropy source, these two keys should be identical:
        priv1 = SigningKey.generate(entropy=not_much_entropy)
        priv2 = SigningKey.generate(entropy=not_much_entropy)
        self.assertEqual(
            hexlify(priv1.get_verifying_key().to_string()),
            hexlify(priv2.get_verifying_key().to_string()),
        )
        # likewise, signatures should be identical. Obviously you'd never
        # want to do this with keys you care about, because the secrecy of
        # the private key depends upon using different random numbers for
        # each signature
        sig1 = priv1.sign(b"data", entropy=not_much_entropy)
        sig2 = priv2.sign(b"data", entropy=not_much_entropy)
        self.assertEqual(hexlify(sig1), hexlify(sig2))

    def assertTruePrivkeysEqual(self, priv1, priv2):
        self.assertEqual(
            priv1.privkey.secret_multiplier, priv2.privkey.secret_multiplier
        )
        self.assertEqual(
            priv1.privkey.public_key.generator,
            priv2.privkey.public_key.generator,
        )

    def test_privkey_creation(self):
        s = b"all the entropy in the entire world, compressed into one line"

        def not_much_entropy(numbytes):
            return s[:numbytes]

        priv1 = SigningKey.generate()
        self.assertEqual(priv1.baselen, NIST192p.baselen)

        priv1 = SigningKey.generate(curve=NIST224p)
        self.assertEqual(priv1.baselen, NIST224p.baselen)

        priv1 = SigningKey.generate(entropy=not_much_entropy)
        self.assertEqual(priv1.baselen, NIST192p.baselen)
        priv2 = SigningKey.generate(entropy=not_much_entropy)
        self.assertEqual(priv2.baselen, NIST192p.baselen)
        self.assertTruePrivkeysEqual(priv1, priv2)

        priv1 = SigningKey.from_secret_exponent(secexp=3)
        self.assertEqual(priv1.baselen, NIST192p.baselen)
        priv2 = SigningKey.from_secret_exponent(secexp=3)
        self.assertTruePrivkeysEqual(priv1, priv2)

        priv1 = SigningKey.from_secret_exponent(secexp=4, curve=NIST224p)
        self.assertEqual(priv1.baselen, NIST224p.baselen)

    def test_privkey_strings(self):
        priv1 = SigningKey.generate()
        s1 = priv1.to_string()
        self.assertEqual(type(s1), binary_type)
        self.assertEqual(len(s1), NIST192p.baselen)
        priv2 = SigningKey.from_string(s1)
        self.assertTruePrivkeysEqual(priv1, priv2)

        s1 = priv1.to_pem()
        self.assertEqual(type(s1), binary_type)
        self.assertTrue(s1.startswith(b"-----BEGIN EC PRIVATE KEY-----"))
        self.assertTrue(s1.strip().endswith(b"-----END EC PRIVATE KEY-----"))
        priv2 = SigningKey.from_pem(s1)
        self.assertTruePrivkeysEqual(priv1, priv2)

        s1 = priv1.to_der()
        self.assertEqual(type(s1), binary_type)
        priv2 = SigningKey.from_der(s1)
        self.assertTruePrivkeysEqual(priv1, priv2)

        priv1 = SigningKey.generate(curve=NIST256p)
        s1 = priv1.to_pem()
        self.assertEqual(type(s1), binary_type)
        self.assertTrue(s1.startswith(b"-----BEGIN EC PRIVATE KEY-----"))
        self.assertTrue(s1.strip().endswith(b"-----END EC PRIVATE KEY-----"))
        priv2 = SigningKey.from_pem(s1)
        self.assertTruePrivkeysEqual(priv1, priv2)

        s1 = priv1.to_der()
        self.assertEqual(type(s1), binary_type)
        priv2 = SigningKey.from_der(s1)
        self.assertTruePrivkeysEqual(priv1, priv2)

    def test_privkey_strings_brainpool(self):
        priv1 = SigningKey.generate(curve=BRAINPOOLP512r1)
        s1 = priv1.to_pem()
        self.assertEqual(type(s1), binary_type)
        self.assertTrue(s1.startswith(b"-----BEGIN EC PRIVATE KEY-----"))
        self.assertTrue(s1.strip().endswith(b"-----END EC PRIVATE KEY-----"))
        priv2 = SigningKey.from_pem(s1)
        self.assertTruePrivkeysEqual(priv1, priv2)

        s1 = priv1.to_der()
        self.assertEqual(type(s1), binary_type)
        priv2 = SigningKey.from_der(s1)
        self.assertTruePrivkeysEqual(priv1, priv2)

    def assertTruePubkeysEqual(self, pub1, pub2):
        self.assertEqual(pub1.pubkey.point, pub2.pubkey.point)
        self.assertEqual(pub1.pubkey.generator, pub2.pubkey.generator)
        self.assertEqual(pub1.curve, pub2.curve)

    def test_pubkey_strings(self):
        priv1 = SigningKey.generate()
        pub1 = priv1.get_verifying_key()
        s1 = pub1.to_string()
        self.assertEqual(type(s1), binary_type)
        self.assertEqual(len(s1), NIST192p.verifying_key_length)
        pub2 = VerifyingKey.from_string(s1)
        self.assertTruePubkeysEqual(pub1, pub2)

        priv1 = SigningKey.generate(curve=NIST256p)
        pub1 = priv1.get_verifying_key()
        s1 = pub1.to_string()
        self.assertEqual(type(s1), binary_type)
        self.assertEqual(len(s1), NIST256p.verifying_key_length)
        pub2 = VerifyingKey.from_string(s1, curve=NIST256p)
        self.assertTruePubkeysEqual(pub1, pub2)

        pub1_der = pub1.to_der()
        self.assertEqual(type(pub1_der), binary_type)
        pub2 = VerifyingKey.from_der(pub1_der)
        self.assertTruePubkeysEqual(pub1, pub2)

        self.assertRaises(
            der.UnexpectedDER, VerifyingKey.from_der, pub1_der + b"junk"
        )
        badpub = VerifyingKey.from_der(pub1_der)

        class FakeGenerator:
            def order(self):
                return 123456789

        class FakeCurveFp:
            def p(self):
                return int(
                    "6525534529039240705020950546962731340"
                    "4541085228058844382513856749047873406763"
                )

        badcurve = Curve(
            "unknown", FakeCurveFp(), FakeGenerator(), (1, 2, 3, 4, 5, 6), None
        )
        badpub.curve = badcurve
        badder = badpub.to_der()
        self.assertRaises(UnknownCurveError, VerifyingKey.from_der, badder)

        pem = pub1.to_pem()
        self.assertEqual(type(pem), binary_type)
        self.assertTrue(pem.startswith(b"-----BEGIN PUBLIC KEY-----"), pem)
        self.assertTrue(pem.strip().endswith(b"-----END PUBLIC KEY-----"), pem)
        pub2 = VerifyingKey.from_pem(pem)
        self.assertTruePubkeysEqual(pub1, pub2)

    def test_pubkey_strings_brainpool(self):
        priv1 = SigningKey.generate(curve=BRAINPOOLP512r1)
        pub1 = priv1.get_verifying_key()
        s1 = pub1.to_string()
        self.assertEqual(type(s1), binary_type)
        self.assertEqual(len(s1), BRAINPOOLP512r1.verifying_key_length)
        pub2 = VerifyingKey.from_string(s1, curve=BRAINPOOLP512r1)
        self.assertTruePubkeysEqual(pub1, pub2)

        pub1_der = pub1.to_der()
        self.assertEqual(type(pub1_der), binary_type)
        pub2 = VerifyingKey.from_der(pub1_der)
        self.assertTruePubkeysEqual(pub1, pub2)

    def test_vk_to_der_with_invalid_point_encoding(self):
        sk = SigningKey.generate()
        vk = sk.verifying_key

        with self.assertRaises(ValueError):
            vk.to_der("raw")

    def test_sk_to_der_with_invalid_point_encoding(self):
        sk = SigningKey.generate()

        with self.assertRaises(ValueError):
            sk.to_der("raw")

    def test_vk_from_der_garbage_after_curve_oid(self):
        type_oid_der = encoded_oid_ecPublicKey
        curve_oid_der = (
            der.encode_oid(*(1, 2, 840, 10045, 3, 1, 1)) + b"garbage"
        )
        enc_type_der = der.encode_sequence(type_oid_der, curve_oid_der)
        point_der = der.encode_bitstring(b"\x00\xff", None)
        to_decode = der.encode_sequence(enc_type_der, point_der)

        with self.assertRaises(der.UnexpectedDER):
            VerifyingKey.from_der(to_decode)

    def test_vk_from_der_invalid_key_type(self):
        type_oid_der = der.encode_oid(*(1, 2, 3))
        curve_oid_der = der.encode_oid(*(1, 2, 840, 10045, 3, 1, 1))
        enc_type_der = der.encode_sequence(type_oid_der, curve_oid_der)
        point_der = der.encode_bitstring(b"\x00\xff", None)
        to_decode = der.encode_sequence(enc_type_der, point_der)

        with self.assertRaises(der.UnexpectedDER):
            VerifyingKey.from_der(to_decode)

    def test_vk_from_der_garbage_after_point_string(self):
        type_oid_der = encoded_oid_ecPublicKey
        curve_oid_der = der.encode_oid(*(1, 2, 840, 10045, 3, 1, 1))
        enc_type_der = der.encode_sequence(type_oid_der, curve_oid_der)
        point_der = der.encode_bitstring(b"\x00\xff", None) + b"garbage"
        to_decode = der.encode_sequence(enc_type_der, point_der)

        with self.assertRaises(der.UnexpectedDER):
            VerifyingKey.from_der(to_decode)

    def test_vk_from_der_invalid_bitstring(self):
        type_oid_der = encoded_oid_ecPublicKey
        curve_oid_der = der.encode_oid(*(1, 2, 840, 10045, 3, 1, 1))
        enc_type_der = der.encode_sequence(type_oid_der, curve_oid_der)
        point_der = der.encode_bitstring(b"\x08\xff", None)
        to_decode = der.encode_sequence(enc_type_der, point_der)

        with self.assertRaises(der.UnexpectedDER):
            VerifyingKey.from_der(to_decode)

    def test_vk_from_der_with_invalid_length_of_encoding(self):
        type_oid_der = encoded_oid_ecPublicKey
        curve_oid_der = der.encode_oid(*(1, 2, 840, 10045, 3, 1, 1))
        enc_type_der = der.encode_sequence(type_oid_der, curve_oid_der)
        point_der = der.encode_bitstring(b"\xff" * 64, 0)
        to_decode = der.encode_sequence(enc_type_der, point_der)

        with self.assertRaises(MalformedPointError):
            VerifyingKey.from_der(to_decode)

    def test_vk_from_der_with_raw_encoding(self):
        type_oid_der = encoded_oid_ecPublicKey
        curve_oid_der = der.encode_oid(*(1, 2, 840, 10045, 3, 1, 1))
        enc_type_der = der.encode_sequence(type_oid_der, curve_oid_der)
        point_der = der.encode_bitstring(b"\xff" * 48, 0)
        to_decode = der.encode_sequence(enc_type_der, point_der)

        with self.assertRaises(der.UnexpectedDER):
            VerifyingKey.from_der(to_decode)

    def test_signature_strings(self):
        priv1 = SigningKey.generate()
        pub1 = priv1.get_verifying_key()
        data = b"data"

        sig = priv1.sign(data)
        self.assertEqual(type(sig), binary_type)
        self.assertEqual(len(sig), NIST192p.signature_length)
        self.assertTrue(pub1.verify(sig, data))

        sig = priv1.sign(data, sigencode=sigencode_strings)
        self.assertEqual(type(sig), tuple)
        self.assertEqual(len(sig), 2)
        self.assertEqual(type(sig[0]), binary_type)
        self.assertEqual(type(sig[1]), binary_type)
        self.assertEqual(len(sig[0]), NIST192p.baselen)
        self.assertEqual(len(sig[1]), NIST192p.baselen)
        self.assertTrue(pub1.verify(sig, data, sigdecode=sigdecode_strings))

        sig_der = priv1.sign(data, sigencode=sigencode_der)
        self.assertEqual(type(sig_der), binary_type)
        self.assertTrue(pub1.verify(sig_der, data, sigdecode=sigdecode_der))

    def test_sigencode_string_canonize_no_change(self):
        r = 12
        s = 400
        order = SECP112r1.order

        new_r, new_s = sigdecode_string(
            sigencode_string_canonize(r, s, order), order
        )

        self.assertEqual(r, new_r)
        self.assertEqual(s, new_s)

    def test_sigencode_string_canonize(self):
        r = 12
        order = SECP112r1.order
        s = order - 10

        new_r, new_s = sigdecode_string(
            sigencode_string_canonize(r, s, order), order
        )

        self.assertEqual(r, new_r)
        self.assertEqual(order - s, new_s)

    def test_sigencode_strings_canonize_no_change(self):
        r = 12
        s = 400
        order = SECP112r1.order

        new_r, new_s = sigdecode_strings(
            sigencode_strings_canonize(r, s, order), order
        )

        self.assertEqual(r, new_r)
        self.assertEqual(s, new_s)

    def test_sigencode_strings_canonize(self):
        r = 12
        order = SECP112r1.order
        s = order - 10

        new_r, new_s = sigdecode_strings(
            sigencode_strings_canonize(r, s, order), order
        )

        self.assertEqual(r, new_r)
        self.assertEqual(order - s, new_s)

    def test_sigencode_der_canonize_no_change(self):
        r = 13
        s = 200
        order = SECP112r1.order

        new_r, new_s = sigdecode_der(
            sigencode_der_canonize(r, s, order), order
        )

        self.assertEqual(r, new_r)
        self.assertEqual(s, new_s)

    def test_sigencode_der_canonize(self):
        r = 13
        order = SECP112r1.order
        s = order - 14

        new_r, new_s = sigdecode_der(
            sigencode_der_canonize(r, s, order), order
        )

        self.assertEqual(r, new_r)
        self.assertEqual(order - s, new_s)

    def test_sigencode_der_canonize_with_close_to_half_order(self):
        r = 13
        order = SECP112r1.order
        s = order // 2 + 1

        regular_encode = sigencode_der(r, s, order)
        canonical_encode = sigencode_der_canonize(r, s, order)

        self.assertNotEqual(regular_encode, canonical_encode)

        new_r, new_s = sigdecode_der(
            sigencode_der_canonize(r, s, order), order
        )

        self.assertEqual(r, new_r)
        self.assertEqual(order - s, new_s)

    def test_sig_decode_strings_with_invalid_count(self):
        with self.assertRaises(MalformedSignature):
            sigdecode_strings([b"one", b"two", b"three"], 0xFF)

    def test_sig_decode_strings_with_wrong_r_len(self):
        with self.assertRaises(MalformedSignature):
            sigdecode_strings([b"one", b"two"], 0xFF)

    def test_sig_decode_strings_with_wrong_s_len(self):
        with self.assertRaises(MalformedSignature):
            sigdecode_strings([b"\xa0", b"\xb0\xff"], 0xFF)

    def test_verify_with_too_long_input(self):
        sk = SigningKey.generate()
        vk = sk.verifying_key

        with self.assertRaises(BadDigestError):
            vk.verify_digest(None, b"\x00" * 128)

    def test_sk_from_secret_exponent_with_wrong_sec_exponent(self):
        with self.assertRaises(MalformedPointError):
            SigningKey.from_secret_exponent(0)

    def test_sk_from_string_with_wrong_len_string(self):
        with self.assertRaises(MalformedPointError):
            SigningKey.from_string(b"\x01")

    def test_sk_from_der_with_junk_after_sequence(self):
        ver_der = der.encode_integer(1)
        to_decode = der.encode_sequence(ver_der) + b"garbage"

        with self.assertRaises(der.UnexpectedDER):
            SigningKey.from_der(to_decode)

    def test_sk_from_der_with_wrong_version(self):
        ver_der = der.encode_integer(0)
        to_decode = der.encode_sequence(ver_der)

        with self.assertRaises(der.UnexpectedDER):
            SigningKey.from_der(to_decode)

    def test_sk_from_der_invalid_const_tag(self):
        ver_der = der.encode_integer(1)
        privkey_der = der.encode_octet_string(b"\x00\xff")
        curve_oid_der = der.encode_oid(*(1, 2, 3))
        const_der = der.encode_constructed(1, curve_oid_der)
        to_decode = der.encode_sequence(
            ver_der, privkey_der, const_der, curve_oid_der
        )

        with self.assertRaises(der.UnexpectedDER):
            SigningKey.from_der(to_decode)

    def test_sk_from_der_garbage_after_privkey_oid(self):
        ver_der = der.encode_integer(1)
        privkey_der = der.encode_octet_string(b"\x00\xff")
        curve_oid_der = der.encode_oid(*(1, 2, 3)) + b"garbage"
        const_der = der.encode_constructed(0, curve_oid_der)
        to_decode = der.encode_sequence(
            ver_der, privkey_der, const_der, curve_oid_der
        )

        with self.assertRaises(der.UnexpectedDER):
            SigningKey.from_der(to_decode)

    def test_sk_from_der_with_short_privkey(self):
        ver_der = der.encode_integer(1)
        privkey_der = der.encode_octet_string(b"\x00\xff")
        curve_oid_der = der.encode_oid(*(1, 2, 840, 10045, 3, 1, 1))
        const_der = der.encode_constructed(0, curve_oid_der)
        to_decode = der.encode_sequence(
            ver_der, privkey_der, const_der, curve_oid_der
        )

        sk = SigningKey.from_der(to_decode)
        self.assertEqual(sk.privkey.secret_multiplier, 255)

    def test_sk_from_p8_der_with_wrong_version(self):
        ver_der = der.encode_integer(2)
        algorithm_der = der.encode_sequence(
            der.encode_oid(1, 2, 840, 10045, 2, 1),
            der.encode_oid(1, 2, 840, 10045, 3, 1, 1),
        )
        privkey_der = der.encode_octet_string(
            der.encode_sequence(
                der.encode_integer(1), der.encode_octet_string(b"\x00\xff")
            )
        )
        to_decode = der.encode_sequence(ver_der, algorithm_der, privkey_der)

        with self.assertRaises(der.UnexpectedDER):
            SigningKey.from_der(to_decode)

    def test_sk_from_p8_der_with_wrong_algorithm(self):
        ver_der = der.encode_integer(1)
        algorithm_der = der.encode_sequence(
            der.encode_oid(1, 2, 3), der.encode_oid(1, 2, 840, 10045, 3, 1, 1)
        )
        privkey_der = der.encode_octet_string(
            der.encode_sequence(
                der.encode_integer(1), der.encode_octet_string(b"\x00\xff")
            )
        )
        to_decode = der.encode_sequence(ver_der, algorithm_der, privkey_der)

        with self.assertRaises(der.UnexpectedDER):
            SigningKey.from_der(to_decode)

    def test_sk_from_p8_der_with_trailing_junk_after_algorithm(self):
        ver_der = der.encode_integer(1)
        algorithm_der = der.encode_sequence(
            der.encode_oid(1, 2, 840, 10045, 2, 1),
            der.encode_oid(1, 2, 840, 10045, 3, 1, 1),
            der.encode_octet_string(b"junk"),
        )
        privkey_der = der.encode_octet_string(
            der.encode_sequence(
                der.encode_integer(1), der.encode_octet_string(b"\x00\xff")
            )
        )
        to_decode = der.encode_sequence(ver_der, algorithm_der, privkey_der)

        with self.assertRaises(der.UnexpectedDER):
            SigningKey.from_der(to_decode)

    def test_sk_from_p8_der_with_trailing_junk_after_key(self):
        ver_der = der.encode_integer(1)
        algorithm_der = der.encode_sequence(
            der.encode_oid(1, 2, 840, 10045, 2, 1),
            der.encode_oid(1, 2, 840, 10045, 3, 1, 1),
        )
        privkey_der = der.encode_octet_string(
            der.encode_sequence(
                der.encode_integer(1), der.encode_octet_string(b"\x00\xff")
            )
            + der.encode_integer(999)
        )
        to_decode = der.encode_sequence(
            ver_der,
            algorithm_der,
            privkey_der,
            der.encode_octet_string(b"junk"),
        )

        with self.assertRaises(der.UnexpectedDER):
            SigningKey.from_der(to_decode)

    def test_sign_with_too_long_hash(self):
        sk = SigningKey.from_secret_exponent(12)

        with self.assertRaises(BadDigestError):
            sk.sign_digest(b"\xff" * 64)

    def test_hashfunc(self):
        sk = SigningKey.generate(curve=NIST256p, hashfunc=hashlib.sha256)
        data = b"security level is 128 bits"
        sig = sk.sign(data)
        vk = VerifyingKey.from_string(
            sk.get_verifying_key().to_string(),
            curve=NIST256p,
            hashfunc=hashlib.sha256,
        )
        self.assertTrue(vk.verify(sig, data))

        sk2 = SigningKey.generate(curve=NIST256p)
        sig2 = sk2.sign(data, hashfunc=hashlib.sha256)
        vk2 = VerifyingKey.from_string(
            sk2.get_verifying_key().to_string(),
            curve=NIST256p,
            hashfunc=hashlib.sha256,
        )
        self.assertTrue(vk2.verify(sig2, data))

        vk3 = VerifyingKey.from_string(
            sk.get_verifying_key().to_string(), curve=NIST256p
        )
        self.assertTrue(vk3.verify(sig, data, hashfunc=hashlib.sha256))

    def test_public_key_recovery(self):
        # Create keys
        curve = BRAINPOOLP160r1

        sk = SigningKey.generate(curve=curve)
        vk = sk.get_verifying_key()

        # Sign a message
        data = b"blahblah"
        signature = sk.sign(data)

        # Recover verifying keys
        recovered_vks = VerifyingKey.from_public_key_recovery(
            signature, data, curve
        )

        # Test if each pk is valid
        for recovered_vk in recovered_vks:
            # Test if recovered vk is valid for the data
            self.assertTrue(recovered_vk.verify(signature, data))

            # Test if properties are equal
            self.assertEqual(vk.curve, recovered_vk.curve)
            self.assertEqual(
                vk.default_hashfunc, recovered_vk.default_hashfunc
            )

        # Test if original vk is the list of recovered keys
        self.assertIn(
            vk.pubkey.point,
            [recovered_vk.pubkey.point for recovered_vk in recovered_vks],
        )

    def test_public_key_recovery_with_custom_hash(self):
        # Create keys
        curve = BRAINPOOLP160r1

        sk = SigningKey.generate(curve=curve, hashfunc=hashlib.sha256)
        vk = sk.get_verifying_key()

        # Sign a message
        data = b"blahblah"
        signature = sk.sign(data)

        # Recover verifying keys
        recovered_vks = VerifyingKey.from_public_key_recovery(
            signature,
            data,
            curve,
            hashfunc=hashlib.sha256,
            allow_truncate=True,
        )

        # Test if each pk is valid
        for recovered_vk in recovered_vks:
            # Test if recovered vk is valid for the data
            self.assertTrue(recovered_vk.verify(signature, data))

            # Test if properties are equal
            self.assertEqual(vk.curve, recovered_vk.curve)
            self.assertEqual(hashlib.sha256, recovered_vk.default_hashfunc)

        # Test if original vk is the list of recovered keys
        self.assertIn(
            vk.pubkey.point,
            [recovered_vk.pubkey.point for recovered_vk in recovered_vks],
        )

    def test_encoding(self):
        sk = SigningKey.from_secret_exponent(123456789)
        vk = sk.verifying_key

        exp = (
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )
        self.assertEqual(vk.to_string(), exp)
        self.assertEqual(vk.to_string("raw"), exp)
        self.assertEqual(vk.to_string("uncompressed"), b"\x04" + exp)
        self.assertEqual(vk.to_string("compressed"), b"\x02" + exp[:24])
        self.assertEqual(vk.to_string("hybrid"), b"\x06" + exp)

    def test_decoding(self):
        sk = SigningKey.from_secret_exponent(123456789)
        vk = sk.verifying_key

        enc = (
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )

        from_raw = VerifyingKey.from_string(enc)
        self.assertEqual(from_raw.pubkey.point, vk.pubkey.point)

        from_uncompressed = VerifyingKey.from_string(b"\x04" + enc)
        self.assertEqual(from_uncompressed.pubkey.point, vk.pubkey.point)

        from_compressed = VerifyingKey.from_string(b"\x02" + enc[:24])
        self.assertEqual(from_compressed.pubkey.point, vk.pubkey.point)

        from_uncompressed = VerifyingKey.from_string(b"\x06" + enc)
        self.assertEqual(from_uncompressed.pubkey.point, vk.pubkey.point)

    def test_uncompressed_decoding_as_only_alowed(self):
        enc = (
            b"\x04"
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )
        vk = VerifyingKey.from_string(enc, valid_encodings=("uncompressed",))
        sk = SigningKey.from_secret_exponent(123456789)

        self.assertEqual(vk, sk.verifying_key)

    def test_raw_decoding_with_blocked_format(self):
        enc = (
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )
        with self.assertRaises(MalformedPointError) as exp:
            VerifyingKey.from_string(enc, valid_encodings=("hybrid",))

        self.assertIn("hybrid", str(exp.exception))

    def test_decoding_with_unknown_format(self):
        with self.assertRaises(ValueError) as e:
            VerifyingKey.from_string(b"", valid_encodings=("raw", "foobar"))

        self.assertIn("Only uncompressed, compressed", str(e.exception))

    def test_uncompressed_decoding_with_blocked_format(self):
        enc = (
            b"\x04"
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )
        with self.assertRaises(MalformedPointError) as exp:
            VerifyingKey.from_string(enc, valid_encodings=("hybrid",))

        self.assertIn("Invalid X9.62 encoding", str(exp.exception))

    def test_hybrid_decoding_with_blocked_format(self):
        enc = (
            b"\x06"
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )
        with self.assertRaises(MalformedPointError) as exp:
            VerifyingKey.from_string(enc, valid_encodings=("uncompressed",))

        self.assertIn("Invalid X9.62 encoding", str(exp.exception))

    def test_hybrid_decoding_with_inconsistent_encoding_and_no_validation(
        self,
    ):
        sk = SigningKey.from_secret_exponent(123456789)
        vk = sk.verifying_key

        enc = vk.to_string("hybrid")
        self.assertEqual(enc[:1], b"\x06")
        enc = b"\x07" + enc[1:]

        b = VerifyingKey.from_string(
            enc, valid_encodings=("hybrid",), validate_point=False
        )

        self.assertEqual(vk, b)

    def test_compressed_decoding_with_blocked_format(self):
        enc = (
            b"\x02"
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )[:25]
        with self.assertRaises(MalformedPointError) as exp:
            VerifyingKey.from_string(enc, valid_encodings=("hybrid", "raw"))

        self.assertIn("(hybrid, raw)", str(exp.exception))

    def test_decoding_with_malformed_uncompressed(self):
        enc = (
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )

        with self.assertRaises(MalformedPointError):
            VerifyingKey.from_string(b"\x02" + enc)

    def test_decoding_with_malformed_compressed(self):
        enc = (
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )

        with self.assertRaises(MalformedPointError):
            VerifyingKey.from_string(b"\x01" + enc[:24])

    def test_decoding_with_inconsistent_hybrid(self):
        enc = (
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )

        with self.assertRaises(MalformedPointError):
            VerifyingKey.from_string(b"\x07" + enc)

    def test_decoding_with_inconsistent_hybrid_odd_point(self):
        sk = SigningKey.from_secret_exponent(123456791)
        vk = sk.verifying_key

        enc = vk.to_string("hybrid")
        self.assertEqual(enc[:1], b"\x07")
        enc = b"\x06" + enc[1:]

        with self.assertRaises(MalformedPointError):
            b = VerifyingKey.from_string(enc, valid_encodings=("hybrid",))

    def test_decoding_with_point_not_on_curve(self):
        enc = (
            b"\x0c\xe0\x1d\xe0d\x1c\x8eS\x8a\xc0\x9eK\xa8x !\xd5\xc2\xc3"
            b"\xfd\xc8\xa0c\xff\xfb\x02\xb9\xc4\x84)\x1a\x0f\x8b\x87\xa4"
            b"z\x8a#\xb5\x97\xecO\xb6\xa0HQ\x89*"
        )

        with self.assertRaises(MalformedPointError):
            VerifyingKey.from_string(enc[:47] + b"\x00")

    def test_decoding_with_point_at_infinity(self):
        # decoding it is unsupported, as it's not necessary to encode it
        with self.assertRaises(MalformedPointError):
            VerifyingKey.from_string(b"\x00")

    def test_not_lying_on_curve(self):
        enc = number_to_string(NIST192p.curve.p(), NIST192p.curve.p() + 1)

        with self.assertRaises(MalformedPointError):
            VerifyingKey.from_string(b"\x02" + enc)

    def test_from_string_with_invalid_curve_too_short_ver_key_len(self):
        # both verifying_key_length and baselen are calculated internally
        # by the Curve constructor, but since we depend on them verify
        # that inconsistent values are detected
        curve = Curve("test", ecdsa.curve_192, ecdsa.generator_192, (1, 2))
        curve.verifying_key_length = 16
        curve.baselen = 32

        with self.assertRaises(MalformedPointError):
            VerifyingKey.from_string(b"\x00" * 16, curve)

    def test_from_string_with_invalid_curve_too_long_ver_key_len(self):
        # both verifying_key_length and baselen are calculated internally
        # by the Curve constructor, but since we depend on them verify
        # that inconsistent values are detected
        curve = Curve("test", ecdsa.curve_192, ecdsa.generator_192, (1, 2))
        curve.verifying_key_length = 16
        curve.baselen = 16

        with self.assertRaises(MalformedPointError):
            VerifyingKey.from_string(b"\x00" * 16, curve)


@pytest.mark.parametrize(
    "val,even", [(i, j) for i in range(256) for j in [True, False]]
)
def test_VerifyingKey_decode_with_small_values(val, even):
    enc = number_to_string(val, NIST192p.order)

    if even:
        enc = b"\x02" + enc
    else:
        enc = b"\x03" + enc

    # small values can both be actual valid public keys and not, verify that
    # only expected exceptions are raised if they are not
    try:
        vk = VerifyingKey.from_string(enc)
        assert isinstance(vk, VerifyingKey)
    except MalformedPointError:
        assert True


params = []
for curve in curves:
    for enc in ["raw", "uncompressed", "compressed", "hybrid"]:
        params.append(
            pytest.param(curve, enc, id="{0}-{1}".format(curve.name, enc))
        )


@pytest.mark.parametrize("curve,encoding", params)
def test_VerifyingKey_encode_decode(curve, encoding):
    sk = SigningKey.generate(curve=curve)
    vk = sk.verifying_key

    encoded = vk.to_string(encoding)

    from_enc = VerifyingKey.from_string(encoded, curve=curve)

    assert vk.pubkey.point == from_enc.pubkey.point


if "--fast" in sys.argv:  # pragma: no cover
    params = [NIST192p, BRAINPOOLP160r1]
else:
    params = curves


@pytest.mark.parametrize("curve", params)
def test_lengths(curve):
    priv = SigningKey.generate(curve=curve)
    pub1 = priv.get_verifying_key()
    pub2 = VerifyingKey.from_string(pub1.to_string(), curve)
    assert pub1.to_string() == pub2.to_string()
    assert len(pub1.to_string()) == curve.verifying_key_length
    sig = priv.sign(b"data")
    assert len(sig) == curve.signature_length


@pytest.mark.slow
class OpenSSL(unittest.TestCase):
    # test interoperability with OpenSSL tools. Note that openssl's ECDSA
    # sign/verify arguments changed between 0.9.8 and 1.0.0: the early
    # versions require "-ecdsa-with-SHA1", the later versions want just
    # "-SHA1" (or to leave out that argument entirely, which means the
    # signature will use some default digest algorithm, probably determined
    # by the key, probably always SHA1).
    #
    # openssl ecparam -name secp224r1 -genkey -out privkey.pem
    # openssl ec -in privkey.pem -text -noout # get the priv/pub keys
    # openssl dgst -ecdsa-with-SHA1 -sign privkey.pem -out data.sig data.txt
    # openssl asn1parse -in data.sig -inform DER
    #  data.sig is 64 bytes, probably 56b plus ASN1 overhead
    # openssl dgst -ecdsa-with-SHA1 -prverify privkey.pem -signature data.sig data.txt ; echo $?
    # openssl ec -in privkey.pem -pubout -out pubkey.pem
    # openssl ec -in privkey.pem -pubout -outform DER -out pubkey.der

    OPENSSL_SUPPORTED_CURVES = set(
        c.split(":")[0].strip()
        for c in run_openssl("ecparam -list_curves").split("\n")
    )

    def get_openssl_messagedigest_arg(self, hash_name):
        v = run_openssl("version")
        # e.g. "OpenSSL 1.0.0 29 Mar 2010", or "OpenSSL 1.0.0a 1 Jun 2010",
        # or "OpenSSL 0.9.8o 01 Jun 2010"
        vs = v.split()[1].split(".")
        if vs >= ["1", "0", "0"]:  # pragma: no cover
            return "-{0}".format(hash_name)
        else:  # pragma: no cover
            return "-ecdsa-with-{0}".format(hash_name)

    # sk: 1:OpenSSL->python  2:python->OpenSSL
    # vk: 3:OpenSSL->python  4:python->OpenSSL
    # sig: 5:OpenSSL->python 6:python->OpenSSL

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp112r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp112r1",
    )
    def test_from_openssl_secp112r1(self):
        return self.do_test_from_openssl(SECP112r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp112r2" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp112r2",
    )
    def test_from_openssl_secp112r2(self):
        return self.do_test_from_openssl(SECP112r2)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp128r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp128r1",
    )
    def test_from_openssl_secp128r1(self):
        return self.do_test_from_openssl(SECP128r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp160r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp160r1",
    )
    def test_from_openssl_secp160r1(self):
        return self.do_test_from_openssl(SECP160r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime192v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime192v1",
    )
    def test_from_openssl_nist192p(self):
        return self.do_test_from_openssl(NIST192p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime192v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime192v1",
    )
    def test_from_openssl_nist192p_sha256(self):
        return self.do_test_from_openssl(NIST192p, "SHA256")

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp224r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp224r1",
    )
    def test_from_openssl_nist224p(self):
        return self.do_test_from_openssl(NIST224p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime256v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime256v1",
    )
    def test_from_openssl_nist256p(self):
        return self.do_test_from_openssl(NIST256p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime256v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime256v1",
    )
    def test_from_openssl_nist256p_sha384(self):
        return self.do_test_from_openssl(NIST256p, "SHA384")

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime256v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime256v1",
    )
    def test_from_openssl_nist256p_sha512(self):
        return self.do_test_from_openssl(NIST256p, "SHA512")

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp384r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp384r1",
    )
    def test_from_openssl_nist384p(self):
        return self.do_test_from_openssl(NIST384p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp521r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp521r1",
    )
    def test_from_openssl_nist521p(self):
        return self.do_test_from_openssl(NIST521p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp256k1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp256k1",
    )
    def test_from_openssl_secp256k1(self):
        return self.do_test_from_openssl(SECP256k1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP160r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP160r1",
    )
    def test_from_openssl_brainpoolp160r1(self):
        return self.do_test_from_openssl(BRAINPOOLP160r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP192r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP192r1",
    )
    def test_from_openssl_brainpoolp192r1(self):
        return self.do_test_from_openssl(BRAINPOOLP192r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP224r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP224r1",
    )
    def test_from_openssl_brainpoolp224r1(self):
        return self.do_test_from_openssl(BRAINPOOLP224r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP256r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP256r1",
    )
    def test_from_openssl_brainpoolp256r1(self):
        return self.do_test_from_openssl(BRAINPOOLP256r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP320r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP320r1",
    )
    def test_from_openssl_brainpoolp320r1(self):
        return self.do_test_from_openssl(BRAINPOOLP320r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP384r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP384r1",
    )
    def test_from_openssl_brainpoolp384r1(self):
        return self.do_test_from_openssl(BRAINPOOLP384r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP512r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP512r1",
    )
    def test_from_openssl_brainpoolp512r1(self):
        return self.do_test_from_openssl(BRAINPOOLP512r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP160t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP160t1",
    )
    def test_from_openssl_brainpoolp160t1(self):
        return self.do_test_from_openssl(BRAINPOOLP160t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP192t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP192t1",
    )
    def test_from_openssl_brainpoolp192t1(self):
        return self.do_test_from_openssl(BRAINPOOLP192t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP224t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP224t1",
    )
    def test_from_openssl_brainpoolp224t1(self):
        return self.do_test_from_openssl(BRAINPOOLP224t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP256t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP256t1",
    )
    def test_from_openssl_brainpoolp256t1(self):
        return self.do_test_from_openssl(BRAINPOOLP256t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP320t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP320t1",
    )
    def test_from_openssl_brainpoolp320t1(self):
        return self.do_test_from_openssl(BRAINPOOLP320t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP384t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP384t1",
    )
    def test_from_openssl_brainpoolp384t1(self):
        return self.do_test_from_openssl(BRAINPOOLP384t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP512t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP512t1",
    )
    def test_from_openssl_brainpoolp512t1(self):
        return self.do_test_from_openssl(BRAINPOOLP512t1)

    def do_test_from_openssl(self, curve, hash_name="SHA1"):
        curvename = curve.openssl_name
        assert curvename
        # OpenSSL: create sk, vk, sign.
        # Python: read vk(3), checksig(5), read sk(1), sign, check
        mdarg = self.get_openssl_messagedigest_arg(hash_name)
        if os.path.isdir("t"):  # pragma: no cover
            shutil.rmtree("t")
        os.mkdir("t")
        run_openssl("ecparam -name %s -genkey -out t/privkey.pem" % curvename)
        run_openssl("ec -in t/privkey.pem -pubout -out t/pubkey.pem")
        data = b"data"
        with open("t/data.txt", "wb") as e:
            e.write(data)
        run_openssl(
            "dgst %s -sign t/privkey.pem -out t/data.sig t/data.txt" % mdarg
        )
        run_openssl(
            "dgst %s -verify t/pubkey.pem -signature t/data.sig t/data.txt"
            % mdarg
        )
        with open("t/pubkey.pem", "rb") as e:
            pubkey_pem = e.read()
        vk = VerifyingKey.from_pem(pubkey_pem)  # 3
        with open("t/data.sig", "rb") as e:
            sig_der = e.read()
        self.assertTrue(
            vk.verify(
                sig_der,
                data,  # 5
                hashfunc=partial(hashlib.new, hash_name),
                sigdecode=sigdecode_der,
            )
        )

        with open("t/privkey.pem") as e:
            fp = e.read()
        sk = SigningKey.from_pem(fp)  # 1
        sig = sk.sign(data, hashfunc=partial(hashlib.new, hash_name))
        self.assertTrue(
            vk.verify(sig, data, hashfunc=partial(hashlib.new, hash_name))
        )

        run_openssl(
            "pkcs8 -topk8 -nocrypt "
            "-in t/privkey.pem -outform pem -out t/privkey-p8.pem"
        )
        with open("t/privkey-p8.pem", "rb") as e:
            privkey_p8_pem = e.read()
        sk_from_p8 = SigningKey.from_pem(privkey_p8_pem)
        self.assertEqual(sk, sk_from_p8)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp112r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp112r1",
    )
    def test_to_openssl_secp112r1(self):
        self.do_test_to_openssl(SECP112r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp112r2" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp112r2",
    )
    def test_to_openssl_secp112r2(self):
        self.do_test_to_openssl(SECP112r2)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp128r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp128r1",
    )
    def test_to_openssl_secp128r1(self):
        self.do_test_to_openssl(SECP128r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp160r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp160r1",
    )
    def test_to_openssl_secp160r1(self):
        self.do_test_to_openssl(SECP160r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime192v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime192v1",
    )
    def test_to_openssl_nist192p(self):
        self.do_test_to_openssl(NIST192p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime192v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime192v1",
    )
    def test_to_openssl_nist192p_sha256(self):
        self.do_test_to_openssl(NIST192p, "SHA256")

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp224r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp224r1",
    )
    def test_to_openssl_nist224p(self):
        self.do_test_to_openssl(NIST224p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime256v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime256v1",
    )
    def test_to_openssl_nist256p(self):
        self.do_test_to_openssl(NIST256p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime256v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime256v1",
    )
    def test_to_openssl_nist256p_sha384(self):
        self.do_test_to_openssl(NIST256p, "SHA384")

    @pytest.mark.slow
    @pytest.mark.skipif(
        "prime256v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime256v1",
    )
    def test_to_openssl_nist256p_sha512(self):
        self.do_test_to_openssl(NIST256p, "SHA512")

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp384r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp384r1",
    )
    def test_to_openssl_nist384p(self):
        self.do_test_to_openssl(NIST384p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp521r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp521r1",
    )
    def test_to_openssl_nist521p(self):
        self.do_test_to_openssl(NIST521p)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "secp256k1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support secp256k1",
    )
    def test_to_openssl_secp256k1(self):
        self.do_test_to_openssl(SECP256k1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP160r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP160r1",
    )
    def test_to_openssl_brainpoolp160r1(self):
        self.do_test_to_openssl(BRAINPOOLP160r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP192r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP192r1",
    )
    def test_to_openssl_brainpoolp192r1(self):
        self.do_test_to_openssl(BRAINPOOLP192r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP224r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP224r1",
    )
    def test_to_openssl_brainpoolp224r1(self):
        self.do_test_to_openssl(BRAINPOOLP224r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP256r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP256r1",
    )
    def test_to_openssl_brainpoolp256r1(self):
        self.do_test_to_openssl(BRAINPOOLP256r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP320r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP320r1",
    )
    def test_to_openssl_brainpoolp320r1(self):
        self.do_test_to_openssl(BRAINPOOLP320r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP384r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP384r1",
    )
    def test_to_openssl_brainpoolp384r1(self):
        self.do_test_to_openssl(BRAINPOOLP384r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP512r1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP512r1",
    )
    def test_to_openssl_brainpoolp512r1(self):
        self.do_test_to_openssl(BRAINPOOLP512r1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP160t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP160t1",
    )
    def test_to_openssl_brainpoolp160t1(self):
        self.do_test_to_openssl(BRAINPOOLP160t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP192t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP192t1",
    )
    def test_to_openssl_brainpoolp192t1(self):
        self.do_test_to_openssl(BRAINPOOLP192t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP224t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP224t1",
    )
    def test_to_openssl_brainpoolp224t1(self):
        self.do_test_to_openssl(BRAINPOOLP224t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP256t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP256t1",
    )
    def test_to_openssl_brainpoolp256t1(self):
        self.do_test_to_openssl(BRAINPOOLP256t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP320t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP320t1",
    )
    def test_to_openssl_brainpoolp320t1(self):
        self.do_test_to_openssl(BRAINPOOLP320t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP384t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP384t1",
    )
    def test_to_openssl_brainpoolp384t1(self):
        self.do_test_to_openssl(BRAINPOOLP384t1)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "brainpoolP512t1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support brainpoolP512t1",
    )
    def test_to_openssl_brainpoolp512t1(self):
        self.do_test_to_openssl(BRAINPOOLP512t1)

    def do_test_to_openssl(self, curve, hash_name="SHA1"):
        curvename = curve.openssl_name
        assert curvename
        # Python: create sk, vk, sign.
        # OpenSSL: read vk(4), checksig(6), read sk(2), sign, check
        mdarg = self.get_openssl_messagedigest_arg(hash_name)
        if os.path.isdir("t"):  # pragma: no cover
            shutil.rmtree("t")
        os.mkdir("t")
        sk = SigningKey.generate(curve=curve)
        vk = sk.get_verifying_key()
        data = b"data"
        with open("t/pubkey.der", "wb") as e:
            e.write(vk.to_der())  # 4
        with open("t/pubkey.pem", "wb") as e:
            e.write(vk.to_pem())  # 4
        sig_der = sk.sign(
            data,
            hashfunc=partial(hashlib.new, hash_name),
            sigencode=sigencode_der,
        )

        with open("t/data.sig", "wb") as e:
            e.write(sig_der)  # 6
        with open("t/data.txt", "wb") as e:
            e.write(data)
        with open("t/baddata.txt", "wb") as e:
            e.write(data + b"corrupt")

        self.assertRaises(
            SubprocessError,
            run_openssl,
            "dgst %s -verify t/pubkey.der -keyform DER -signature t/data.sig t/baddata.txt"
            % mdarg,
        )
        run_openssl(
            "dgst %s -verify t/pubkey.der -keyform DER -signature t/data.sig t/data.txt"
            % mdarg
        )

        with open("t/privkey.pem", "wb") as e:
            e.write(sk.to_pem())  # 2
        run_openssl(
            "dgst %s -sign t/privkey.pem -out t/data.sig2 t/data.txt" % mdarg
        )
        run_openssl(
            "dgst %s -verify t/pubkey.pem -signature t/data.sig2 t/data.txt"
            % mdarg
        )

        with open("t/privkey-explicit.pem", "wb") as e:
            e.write(sk.to_pem(curve_parameters_encoding="explicit"))
        run_openssl(
            "dgst %s -sign t/privkey-explicit.pem -out t/data.sig2 t/data.txt"
            % mdarg
        )
        run_openssl(
            "dgst %s -verify t/pubkey.pem -signature t/data.sig2 t/data.txt"
            % mdarg
        )

        with open("t/privkey-p8.pem", "wb") as e:
            e.write(sk.to_pem(format="pkcs8"))
        run_openssl(
            "dgst %s -sign t/privkey-p8.pem -out t/data.sig3 t/data.txt"
            % mdarg
        )
        run_openssl(
            "dgst %s -verify t/pubkey.pem -signature t/data.sig3 t/data.txt"
            % mdarg
        )

        with open("t/privkey-p8-explicit.pem", "wb") as e:
            e.write(
                sk.to_pem(format="pkcs8", curve_parameters_encoding="explicit")
            )
        run_openssl(
            "dgst %s -sign t/privkey-p8-explicit.pem -out t/data.sig3 t/data.txt"
            % mdarg
        )
        run_openssl(
            "dgst %s -verify t/pubkey.pem -signature t/data.sig3 t/data.txt"
            % mdarg
        )

    OPENSSL_SUPPORTED_TYPES = set()
    try:
        if "-rawin" in run_openssl("pkeyutl -help"):
            OPENSSL_SUPPORTED_TYPES = set(  # pragma: no branch
                c.lower()
                for c in ("ED25519", "ED448")
                if c in run_openssl("list -public-key-methods")
            )
    except SubprocessError:  # pragma: no cover
        pass

    def do_eddsa_test_to_openssl(self, curve):
        if os.path.isdir("t"):
            shutil.rmtree("t")
        os.mkdir("t")

        sk = SigningKey.generate(curve=curve)
        vk = sk.get_verifying_key()

        data = b"data"
        with open("t/pubkey.der", "wb") as e:
            e.write(vk.to_der())
        with open("t/pubkey.pem", "wb") as e:
            e.write(vk.to_pem())

        sig = sk.sign(data)

        with open("t/data.sig", "wb") as e:
            e.write(sig)
        with open("t/data.txt", "wb") as e:
            e.write(data)
        with open("t/baddata.txt", "wb") as e:
            e.write(data + b"corrupt")

        with self.assertRaises(SubprocessError):
            run_openssl(
                "pkeyutl -verify -pubin -inkey t/pubkey.pem -rawin "
                "-in t/baddata.txt -sigfile t/data.sig"
            )
        run_openssl(
            "pkeyutl -verify -pubin -inkey t/pubkey.pem -rawin "
            "-in t/data.txt -sigfile t/data.sig"
        )

        shutil.rmtree("t")

    # in practice at least OpenSSL 3.0.0 is needed to make EdDSA signatures
    # earlier versions support EdDSA only in X.509 certificates
    @pytest.mark.slow
    @pytest.mark.skipif(
        "ed25519" not in OPENSSL_SUPPORTED_TYPES,
        reason="system openssl does not support signing with Ed25519",
    )
    def test_to_openssl_ed25519(self):
        return self.do_eddsa_test_to_openssl(Ed25519)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "ed448" not in OPENSSL_SUPPORTED_TYPES,
        reason="system openssl does not support signing with Ed448",
    )
    def test_to_openssl_ed448(self):
        return self.do_eddsa_test_to_openssl(Ed448)

    def do_eddsa_test_from_openssl(self, curve):
        curvename = curve.name

        if os.path.isdir("t"):
            shutil.rmtree("t")
        os.mkdir("t")

        data = b"data"

        run_openssl(
            "genpkey -algorithm {0} -outform PEM -out t/privkey.pem".format(
                curvename
            )
        )
        run_openssl(
            "pkey -outform PEM -pubout -in t/privkey.pem -out t/pubkey.pem"
        )

        with open("t/data.txt", "wb") as e:
            e.write(data)
        run_openssl(
            "pkeyutl -sign -inkey t/privkey.pem "
            "-rawin -in t/data.txt -out t/data.sig"
        )

        with open("t/data.sig", "rb") as e:
            sig = e.read()
        with open("t/pubkey.pem", "rb") as e:
            vk = VerifyingKey.from_pem(e.read())

        self.assertIs(vk.curve, curve)

        vk.verify(sig, data)

        shutil.rmtree("t")

    @pytest.mark.slow
    @pytest.mark.skipif(
        "ed25519" not in OPENSSL_SUPPORTED_TYPES,
        reason="system openssl does not support signing with Ed25519",
    )
    def test_from_openssl_ed25519(self):
        return self.do_eddsa_test_from_openssl(Ed25519)

    @pytest.mark.slow
    @pytest.mark.skipif(
        "ed448" not in OPENSSL_SUPPORTED_TYPES,
        reason="system openssl does not support signing with Ed448",
    )
    def test_from_openssl_ed448(self):
        return self.do_eddsa_test_from_openssl(Ed448)


class TooSmallCurve(unittest.TestCase):
    OPENSSL_SUPPORTED_CURVES = set(
        c.split(":")[0].strip()
        for c in run_openssl("ecparam -list_curves").split("\n")
    )

    @pytest.mark.skipif(
        "prime192v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime192v1",
    )
    def test_sign_too_small_curve_dont_allow_truncate_raises(self):
        sk = SigningKey.generate(curve=NIST192p)
        data = b"data"
        with self.assertRaises(BadDigestError):
            sk.sign(
                data,
                hashfunc=partial(hashlib.new, "SHA256"),
                sigencode=sigencode_der,
                allow_truncate=False,
            )

    @pytest.mark.skipif(
        "prime192v1" not in OPENSSL_SUPPORTED_CURVES,
        reason="system openssl does not support prime192v1",
    )
    def test_verify_too_small_curve_dont_allow_truncate_raises(self):
        sk = SigningKey.generate(curve=NIST192p)
        vk = sk.get_verifying_key()
        data = b"data"
        sig_der = sk.sign(
            data,
            hashfunc=partial(hashlib.new, "SHA256"),
            sigencode=sigencode_der,
            allow_truncate=True,
        )
        with self.assertRaises(BadDigestError):
            vk.verify(
                sig_der,
                data,
                hashfunc=partial(hashlib.new, "SHA256"),
                sigdecode=sigdecode_der,
                allow_truncate=False,
            )


class DER(unittest.TestCase):
    def test_integer(self):
        self.assertEqual(der.encode_integer(0), b"\x02\x01\x00")
        self.assertEqual(der.encode_integer(1), b"\x02\x01\x01")
        self.assertEqual(der.encode_integer(127), b"\x02\x01\x7f")
        self.assertEqual(der.encode_integer(128), b"\x02\x02\x00\x80")
        self.assertEqual(der.encode_integer(256), b"\x02\x02\x01\x00")
        # self.assertEqual(der.encode_integer(-1), b"\x02\x01\xff")

        def s(n):
            return der.remove_integer(der.encode_integer(n) + b"junk")

        self.assertEqual(s(0), (0, b"junk"))
        self.assertEqual(s(1), (1, b"junk"))
        self.assertEqual(s(127), (127, b"junk"))
        self.assertEqual(s(128), (128, b"junk"))
        self.assertEqual(s(256), (256, b"junk"))
        self.assertEqual(
            s(1234567890123456789012345678901234567890),
            (1234567890123456789012345678901234567890, b"junk"),
        )

    def test_number(self):
        self.assertEqual(der.encode_number(0), b"\x00")
        self.assertEqual(der.encode_number(127), b"\x7f")
        self.assertEqual(der.encode_number(128), b"\x81\x00")
        self.assertEqual(der.encode_number(3 * 128 + 7), b"\x83\x07")
        # self.assertEqual(der.read_number("\x81\x9b" + "more"), (155, 2))
        # self.assertEqual(der.encode_number(155), b"\x81\x9b")
        for n in (0, 1, 2, 127, 128, 3 * 128 + 7, 840, 10045):  # , 155):
            x = der.encode_number(n) + b"more"
            n1, llen = der.read_number(x)
            self.assertEqual(n1, n)
            self.assertEqual(x[llen:], b"more")

    def test_length(self):
        self.assertEqual(der.encode_length(0), b"\x00")
        self.assertEqual(der.encode_length(127), b"\x7f")
        self.assertEqual(der.encode_length(128), b"\x81\x80")
        self.assertEqual(der.encode_length(255), b"\x81\xff")
        self.assertEqual(der.encode_length(256), b"\x82\x01\x00")
        self.assertEqual(der.encode_length(3 * 256 + 7), b"\x82\x03\x07")
        self.assertEqual(der.read_length(b"\x81\x9b" + b"more"), (155, 2))
        self.assertEqual(der.encode_length(155), b"\x81\x9b")
        for n in (0, 1, 2, 127, 128, 255, 256, 3 * 256 + 7, 155):
            x = der.encode_length(n) + b"more"
            n1, llen = der.read_length(x)
            self.assertEqual(n1, n)
            self.assertEqual(x[llen:], b"more")

    def test_sequence(self):
        x = der.encode_sequence(b"ABC", b"DEF") + b"GHI"
        self.assertEqual(x, b"\x30\x06ABCDEFGHI")
        x1, rest = der.remove_sequence(x)
        self.assertEqual(x1, b"ABCDEF")
        self.assertEqual(rest, b"GHI")

    def test_constructed(self):
        x = der.encode_constructed(0, NIST224p.encoded_oid)
        self.assertEqual(hexlify(x), b"a007" + b"06052b81040021")
        x = der.encode_constructed(1, unhexlify(b"0102030a0b0c"))
        self.assertEqual(hexlify(x), b"a106" + b"0102030a0b0c")


class Util(unittest.TestCase):
    @pytest.mark.slow
    def test_trytryagain(self):
        tta = util.randrange_from_seed__trytryagain
        for i in range(1000):
            seed = "seed-%d" % i
            for order in (
                2**8 - 2,
                2**8 - 1,
                2**8,
                2**8 + 1,
                2**8 + 2,
                2**16 - 1,
                2**16 + 1,
            ):
                n = tta(seed, order)
                self.assertTrue(1 <= n < order, (1, n, order))
        # this trytryagain *does* provide long-term stability
        self.assertEqual(
            ("%x" % (tta("seed", NIST224p.order))).encode(),
            b"6fa59d73bf0446ae8743cf748fc5ac11d5585a90356417e97155c3bc",
        )

    def test_trytryagain_single(self):
        tta = util.randrange_from_seed__trytryagain
        order = 2**8 - 2
        seed = b"text"
        n = tta(seed, order)
        # known issue: https://github.com/warner/python-ecdsa/issues/221
        if sys.version_info < (3, 0):  # pragma: no branch
            self.assertEqual(n, 228)
        else:  # pragma: no branch
            self.assertEqual(n, 18)

    @settings(**HYP_SETTINGS)
    @given(st.integers(min_value=0, max_value=10**200))
    def test_randrange(self, i):
        # util.randrange does not provide long-term stability: we might
        # change the algorithm in the future.
        entropy = util.PRNG("seed-%d" % i)
        for order in (
            2**8 - 2,
            2**8 - 1,
            2**8,
            2**16 - 1,
            2**16 + 1,
        ):
            # that oddball 2**16+1 takes half our runtime
            n = util.randrange(order, entropy=entropy)
            self.assertTrue(1 <= n < order, (1, n, order))

    def OFF_test_prove_uniformity(self):  # pragma: no cover
        order = 2**8 - 2
        counts = dict([(i, 0) for i in range(1, order)])
        assert 0 not in counts
        assert order not in counts
        for i in range(1000000):
            seed = "seed-%d" % i
            n = util.randrange_from_seed__trytryagain(seed, order)
            counts[n] += 1
        # this technique should use the full range
        self.assertTrue(counts[order - 1])
        for i in range(1, order):
            print("%3d: %s" % (i, "*" * (counts[i] // 100)))


class RFC6979(unittest.TestCase):
    # https://tools.ietf.org/html/rfc6979#appendix-A.1
    def _do(self, generator, secexp, hsh, hash_func, expected):
        actual = rfc6979.generate_k(generator.order(), secexp, hash_func, hsh)
        self.assertEqual(expected, actual)

    def test_SECP256k1(self):
        """RFC doesn't contain test vectors for SECP256k1 used in bitcoin.
        This vector has been computed by Golang reference implementation instead."""
        self._do(
            generator=SECP256k1.generator,
            secexp=int("9d0219792467d7d37b4d43298a7d0c05", 16),
            hsh=hashlib.sha256(b"sample").digest(),
            hash_func=hashlib.sha256,
            expected=int(
                "8fa1f95d514760e498f28957b824ee6ec39ed64826ff4fecc2b5739ec45b91cd",
                16,
            ),
        )

    def test_SECP256k1_2(self):
        self._do(
            generator=SECP256k1.generator,
            secexp=int(
                "cca9fbcc1b41e5a95d369eaa6ddcff73b61a4efaa279cfc6567e8daa39cbaf50",
                16,
            ),
            hsh=hashlib.sha256(b"sample").digest(),
            hash_func=hashlib.sha256,
            expected=int(
                "2df40ca70e639d89528a6b670d9d48d9165fdc0febc0974056bdce192b8e16a3",
                16,
            ),
        )

    def test_SECP256k1_3(self):
        self._do(
            generator=SECP256k1.generator,
            secexp=0x1,
            hsh=hashlib.sha256(b"Satoshi Nakamoto").digest(),
            hash_func=hashlib.sha256,
            expected=0x8F8A276C19F4149656B280621E358CCE24F5F52542772691EE69063B74F15D15,
        )

    def test_SECP256k1_4(self):
        self._do(
            generator=SECP256k1.generator,
            secexp=0x1,
            hsh=hashlib.sha256(
                b"All those moments will be lost in time, like tears in rain. Time to die..."
            ).digest(),
            hash_func=hashlib.sha256,
            expected=0x38AA22D72376B4DBC472E06C3BA403EE0A394DA63FC58D88686C611ABA98D6B3,
        )

    def test_SECP256k1_5(self):
        self._do(
            generator=SECP256k1.generator,
            secexp=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140,
            hsh=hashlib.sha256(b"Satoshi Nakamoto").digest(),
            hash_func=hashlib.sha256,
            expected=0x33A19B60E25FB6F4435AF53A3D42D493644827367E6453928554F43E49AA6F90,
        )

    def test_SECP256k1_6(self):
        self._do(
            generator=SECP256k1.generator,
            secexp=0xF8B8AF8CE3C7CCA5E300D33939540C10D45CE001B8F252BFBC57BA0342904181,
            hsh=hashlib.sha256(b"Alan Turing").digest(),
            hash_func=hashlib.sha256,
            expected=0x525A82B70E67874398067543FD84C83D30C175FDC45FDEEE082FE13B1D7CFDF1,
        )

    def test_1(self):
        # Basic example of the RFC, it also tests 'try-try-again' from Step H of rfc6979
        self._do(
            generator=Point(
                None,
                0,
                0,
                int("4000000000000000000020108A2E0CC0D99F8A5EF", 16),
            ),
            secexp=int("09A4D6792295A7F730FC3F2B49CBC0F62E862272F", 16),
            hsh=unhexlify(
                b"AF2BDBE1AA9B6EC1E2ADE1D694F41FC71A831D0268E9891562113D8A62ADD1BF"
            ),
            hash_func=hashlib.sha256,
            expected=int("23AF4074C90A02B3FE61D286D5C87F425E6BDD81B", 16),
        )

    def test_2(self):
        self._do(
            generator=NIST192p.generator,
            secexp=int("6FAB034934E4C0FC9AE67F5B5659A9D7D1FEFD187EE09FD4", 16),
            hsh=hashlib.sha1(b"sample").digest(),
            hash_func=hashlib.sha1,
            expected=int(
                "37D7CA00D2C7B0E5E412AC03BD44BA837FDD5B28CD3B0021", 16
            ),
        )

    def test_3(self):
        self._do(
            generator=NIST192p.generator,
            secexp=int("6FAB034934E4C0FC9AE67F5B5659A9D7D1FEFD187EE09FD4", 16),
            hsh=hashlib.sha256(b"sample").digest(),
            hash_func=hashlib.sha256,
            expected=int(
                "32B1B6D7D42A05CB449065727A84804FB1A3E34D8F261496", 16
            ),
        )

    def test_4(self):
        self._do(
            generator=NIST192p.generator,
            secexp=int("6FAB034934E4C0FC9AE67F5B5659A9D7D1FEFD187EE09FD4", 16),
            hsh=hashlib.sha512(b"sample").digest(),
            hash_func=hashlib.sha512,
            expected=int(
                "A2AC7AB055E4F20692D49209544C203A7D1F2C0BFBC75DB1", 16
            ),
        )

    def test_5(self):
        self._do(
            generator=NIST192p.generator,
            secexp=int("6FAB034934E4C0FC9AE67F5B5659A9D7D1FEFD187EE09FD4", 16),
            hsh=hashlib.sha1(b"test").digest(),
            hash_func=hashlib.sha1,
            expected=int(
                "D9CF9C3D3297D3260773A1DA7418DB5537AB8DD93DE7FA25", 16
            ),
        )

    def test_6(self):
        self._do(
            generator=NIST192p.generator,
            secexp=int("6FAB034934E4C0FC9AE67F5B5659A9D7D1FEFD187EE09FD4", 16),
            hsh=hashlib.sha256(b"test").digest(),
            hash_func=hashlib.sha256,
            expected=int(
                "5C4CE89CF56D9E7C77C8585339B006B97B5F0680B4306C6C", 16
            ),
        )

    def test_7(self):
        self._do(
            generator=NIST192p.generator,
            secexp=int("6FAB034934E4C0FC9AE67F5B5659A9D7D1FEFD187EE09FD4", 16),
            hsh=hashlib.sha512(b"test").digest(),
            hash_func=hashlib.sha512,
            expected=int(
                "0758753A5254759C7CFBAD2E2D9B0792EEE44136C9480527", 16
            ),
        )

    def test_8(self):
        self._do(
            generator=NIST521p.generator,
            secexp=int(
                "0FAD06DAA62BA3B25D2FB40133DA757205DE67F5BB0018FEE8C86E1B68C7E75CAA896EB32F1F47C70855836A6D16FCC1466F6D8FBEC67DB89EC0C08B0E996B83538",
                16,
            ),
            hsh=hashlib.sha1(b"sample").digest(),
            hash_func=hashlib.sha1,
            expected=int(
                "089C071B419E1C2820962321787258469511958E80582E95D8378E0C2CCDB3CB42BEDE42F50E3FA3C71F5A76724281D31D9C89F0F91FC1BE4918DB1C03A5838D0F9",
                16,
            ),
        )

    def test_9(self):
        self._do(
            generator=NIST521p.generator,
            secexp=int(
                "0FAD06DAA62BA3B25D2FB40133DA757205DE67F5BB0018FEE8C86E1B68C7E75CAA896EB32F1F47C70855836A6D16FCC1466F6D8FBEC67DB89EC0C08B0E996B83538",
                16,
            ),
            hsh=hashlib.sha256(b"sample").digest(),
            hash_func=hashlib.sha256,
            expected=int(
                "0EDF38AFCAAECAB4383358B34D67C9F2216C8382AAEA44A3DAD5FDC9C32575761793FEF24EB0FC276DFC4F6E3EC476752F043CF01415387470BCBD8678ED2C7E1A0",
                16,
            ),
        )

    def test_10(self):
        self._do(
            generator=NIST521p.generator,
            secexp=int(
                "0FAD06DAA62BA3B25D2FB40133DA757205DE67F5BB0018FEE8C86E1B68C7E75CAA896EB32F1F47C70855836A6D16FCC1466F6D8FBEC67DB89EC0C08B0E996B83538",
                16,
            ),
            hsh=hashlib.sha512(b"test").digest(),
            hash_func=hashlib.sha512,
            expected=int(
                "16200813020EC986863BEDFC1B121F605C1215645018AEA1A7B215A564DE9EB1B38A67AA1128B80CE391C4FB71187654AAA3431027BFC7F395766CA988C964DC56D",
                16,
            ),
        )


class ECDH(unittest.TestCase):
    def _do(self, curve, generator, dA, x_qA, y_qA, dB, x_qB, y_qB, x_Z, y_Z):
        qA = dA * generator
        qB = dB * generator
        Z = dA * qB
        self.assertEqual(Point(curve, x_qA, y_qA), qA)
        self.assertEqual(Point(curve, x_qB, y_qB), qB)
        self.assertTrue(
            (dA * qB)
            == (dA * dB * generator)
            == (dB * dA * generator)
            == (dB * qA)
        )
        self.assertEqual(Point(curve, x_Z, y_Z), Z)


class RFC6932(ECDH):
    # https://tools.ietf.org/html/rfc6932#appendix-A.1

    def test_brainpoolP224r1(self):
        self._do(
            curve=curve_brainpoolp224r1,
            generator=BRAINPOOLP224r1.generator,
            dA=int(
                "7C4B7A2C8A4BAD1FBB7D79CC0955DB7C6A4660CA64CC4778159B495E", 16
            ),
            x_qA=int(
                "B104A67A6F6E85E14EC1825E1539E8ECDBBF584922367DD88C6BDCF2", 16
            ),
            y_qA=int(
                "46D782E7FDB5F60CD8404301AC5949C58EDB26BC68BA07695B750A94", 16
            ),
            dB=int(
                "63976D4AAE6CD0F6DD18DEFEF55D96569D0507C03E74D6486FFA28FB", 16
            ),
            x_qB=int(
                "2A97089A9296147B71B21A4B574E1278245B536F14D8C2B9D07A874E", 16
            ),
            y_qB=int(
                "9B900D7C77A709A797276B8CA1BA61BB95B546FC29F862E44D59D25B", 16
            ),
            x_Z=int(
                "312DFD98783F9FB77B9704945A73BEB6DCCBE3B65D0F967DCAB574EB", 16
            ),
            y_Z=int(
                "6F800811D64114B1C48C621AB3357CF93F496E4238696A2A012B3C98", 16
            ),
        )

    def test_brainpoolP256r1(self):
        self._do(
            curve=curve_brainpoolp256r1,
            generator=BRAINPOOLP256r1.generator,
            dA=int(
                "041EB8B1E2BC681BCE8E39963B2E9FC415B05283313DD1A8BCC055F11AE"
                "49699",
                16,
            ),
            x_qA=int(
                "78028496B5ECAAB3C8B6C12E45DB1E02C9E4D26B4113BC4F015F60C5C"
                "CC0D206",
                16,
            ),
            y_qA=int(
                "A2AE1762A3831C1D20F03F8D1E3C0C39AFE6F09B4D44BBE80CD100987"
                "B05F92B",
                16,
            ),
            dB=int(
                "06F5240EACDB9837BC96D48274C8AA834B6C87BA9CC3EEDD81F99A16B8D"
                "804D3",
                16,
            ),
            x_qB=int(
                "8E07E219BA588916C5B06AA30A2F464C2F2ACFC1610A3BE2FB240B635"
                "341F0DB",
                16,
            ),
            y_qB=int(
                "148EA1D7D1E7E54B9555B6C9AC90629C18B63BEE5D7AA6949EBBF47B2"
                "4FDE40D",
                16,
            ),
            x_Z=int(
                "05E940915549E9F6A4A75693716E37466ABA79B4BF2919877A16DD2CC2"
                "E23708",
                16,
            ),
            y_Z=int(
                "6BC23B6702BC5A019438CEEA107DAAD8B94232FFBBC350F3B137628FE6"
                "FD134C",
                16,
            ),
        )

    @pytest.mark.slow
    def test_brainpoolP384r1(self):
        self._do(
            curve=curve_brainpoolp384r1,
            generator=BRAINPOOLP384r1.generator,
            dA=int(
                "014EC0755B78594BA47FB0A56F6173045B4331E74BA1A6F47322E70D79D"
                "828D97E095884CA72B73FDABD5910DF0FA76A",
                16,
            ),
            x_qA=int(
                "45CB26E4384DAF6FB776885307B9A38B7AD1B5C692E0C32F012533277"
                "8F3B8D3F50CA358099B30DEB5EE69A95C058B4E",
                16,
            ),
            y_qA=int(
                "8173A1C54AFFA7E781D0E1E1D12C0DC2B74F4DF58E4A4E3AF7026C5D3"
                "2DC530A2CD89C859BB4B4B768497F49AB8CC859",
                16,
            ),
            dB=int(
                "6B461CB79BD0EA519A87D6828815D8CE7CD9B3CAA0B5A8262CBCD550A01"
                "5C90095B976F3529957506E1224A861711D54",
                16,
            ),
            x_qB=int(
                "01BF92A92EE4BE8DED1A911125C209B03F99E3161CFCC986DC7711383"
                "FC30AF9CE28CA3386D59E2C8D72CE1E7B4666E8",
                16,
            ),
            y_qB=int(
                "3289C4A3A4FEE035E39BDB885D509D224A142FF9FBCC5CFE5CCBB3026"
                "8EE47487ED8044858D31D848F7A95C635A347AC",
                16,
            ),
            x_Z=int(
                "04CC4FF3DCCCB07AF24E0ACC529955B36D7C807772B92FCBE48F3AFE9A"
                "2F370A1F98D3FA73FD0C0747C632E12F1423EC",
                16,
            ),
            y_Z=int(
                "7F465F90BD69AFB8F828A214EB9716D66ABC59F17AF7C75EE7F1DE22AB"
                "5D05085F5A01A9382D05BF72D96698FE3FF64E",
                16,
            ),
        )

    @pytest.mark.slow
    def test_brainpoolP512r1(self):
        self._do(
            curve=curve_brainpoolp512r1,
            generator=BRAINPOOLP512r1.generator,
            dA=int(
                "636B6BE0482A6C1C41AA7AE7B245E983392DB94CECEA2660A379CFE1595"
                "59E357581825391175FC195D28BAC0CF03A7841A383B95C262B98378287"
                "4CCE6FE333",
                16,
            ),
            x_qA=int(
                "0562E68B9AF7CBFD5565C6B16883B777FF11C199161ECC427A39D17EC"
                "2166499389571D6A994977C56AD8252658BA8A1B72AE42F4FB7532151"
                "AFC3EF0971CCDA",
                16,
            ),
            y_qA=int(
                "A7CA2D8191E21776A89860AFBC1F582FAA308D551C1DC6133AF9F9C3C"
                "AD59998D70079548140B90B1F311AFB378AA81F51B275B2BE6B7DEE97"
                "8EFC7343EA642E",
                16,
            ),
            dB=int(
                "0AF4E7F6D52EDD52907BB8DBAB3992A0BB696EC10DF11892FF205B66D38"
                "1ECE72314E6A6EA079CEA06961DBA5AE6422EF2E9EE803A1F236FB96A17"
                "99B86E5C8B",
                16,
            ),
            x_qB=int(
                "5A7954E32663DFF11AE24712D87419F26B708AC2B92877D6BFEE2BFC4"
                "3714D89BBDB6D24D807BBD3AEB7F0C325F862E8BADE4F74636B97EAAC"
                "E739E11720D323",
                16,
            ),
            y_qB=int(
                "96D14621A9283A1BED84DE8DD64836B2C0758B11441179DC0C54C0D49"
                "A47C03807D171DD544B72CAAEF7B7CE01C7753E2CAD1A861ECA55A719"
                "54EE1BA35E04BE",
                16,
            ),
            x_Z=int(
                "1EE8321A4BBF93B9CF8921AB209850EC9B7066D1984EF08C2BB7232362"
                "08AC8F1A483E79461A00E0D5F6921CE9D360502F85C812BEDEE23AC5B2"
                "10E5811B191E",
                16,
            ),
            y_Z=int(
                "2632095B7B936174B41FD2FAF369B1D18DCADEED7E410A7E251F083109"
                "7C50D02CFED02607B6A2D5ADB4C0006008562208631875B58B54ECDA5A"
                "4F9FE9EAABA6",
                16,
            ),
        )


class RFC7027(ECDH):
    # https://tools.ietf.org/html/rfc7027#appendix-A

    def test_brainpoolP256r1(self):
        self._do(
            curve=curve_brainpoolp256r1,
            generator=BRAINPOOLP256r1.generator,
            dA=int(
                "81DB1EE100150FF2EA338D708271BE38300CB54241D79950F77B0630398"
                "04F1D",
                16,
            ),
            x_qA=int(
                "44106E913F92BC02A1705D9953A8414DB95E1AAA49E81D9E85F929A8E"
                "3100BE5",
                16,
            ),
            y_qA=int(
                "8AB4846F11CACCB73CE49CBDD120F5A900A69FD32C272223F789EF10E"
                "B089BDC",
                16,
            ),
            dB=int(
                "55E40BC41E37E3E2AD25C3C6654511FFA8474A91A0032087593852D3E7D"
                "76BD3",
                16,
            ),
            x_qB=int(
                "8D2D688C6CF93E1160AD04CC4429117DC2C41825E1E9FCA0ADDD34E6F"
                "1B39F7B",
                16,
            ),
            y_qB=int(
                "990C57520812BE512641E47034832106BC7D3E8DD0E4C7F1136D70065"
                "47CEC6A",
                16,
            ),
            x_Z=int(
                "89AFC39D41D3B327814B80940B042590F96556EC91E6AE7939BCE31F3A"
                "18BF2B",
                16,
            ),
            y_Z=int(
                "49C27868F4ECA2179BFD7D59B1E3BF34C1DBDE61AE12931648F43E5963"
                "2504DE",
                16,
            ),
        )

    @pytest.mark.slow
    def test_brainpoolP384r1(self):
        self._do(
            curve=curve_brainpoolp384r1,
            generator=BRAINPOOLP384r1.generator,
            dA=int(
                "1E20F5E048A5886F1F157C74E91BDE2B98C8B52D58E5003D57053FC4B0B"
                "D65D6F15EB5D1EE1610DF870795143627D042",
                16,
            ),
            x_qA=int(
                "68B665DD91C195800650CDD363C625F4E742E8134667B767B1B476793"
                "588F885AB698C852D4A6E77A252D6380FCAF068",
                16,
            ),
            y_qA=int(
                "55BC91A39C9EC01DEE36017B7D673A931236D2F1F5C83942D049E3FA2"
                "0607493E0D038FF2FD30C2AB67D15C85F7FAA59",
                16,
            ),
            dB=int(
                "032640BC6003C59260F7250C3DB58CE647F98E1260ACCE4ACDA3DD869F7"
                "4E01F8BA5E0324309DB6A9831497ABAC96670",
                16,
            ),
            x_qB=int(
                "4D44326F269A597A5B58BBA565DA5556ED7FD9A8A9EB76C25F46DB69D"
                "19DC8CE6AD18E404B15738B2086DF37E71D1EB4",
                16,
            ),
            y_qB=int(
                "62D692136DE56CBE93BF5FA3188EF58BC8A3A0EC6C1E151A21038A42E"
                "9185329B5B275903D192F8D4E1F32FE9CC78C48",
                16,
            ),
            x_Z=int(
                "0BD9D3A7EA0B3D519D09D8E48D0785FB744A6B355E6304BC51C229FBBC"
                "E239BBADF6403715C35D4FB2A5444F575D4F42",
                16,
            ),
            y_Z=int(
                "0DF213417EBE4D8E40A5F76F66C56470C489A3478D146DECF6DF0D94BA"
                "E9E598157290F8756066975F1DB34B2324B7BD",
                16,
            ),
        )

    @pytest.mark.slow
    def test_brainpoolP512r1(self):
        self._do(
            curve=curve_brainpoolp512r1,
            generator=BRAINPOOLP512r1.generator,
            dA=int(
                "16302FF0DBBB5A8D733DAB7141C1B45ACBC8715939677F6A56850A38BD8"
                "7BD59B09E80279609FF333EB9D4C061231FB26F92EEB04982A5F1D1764C"
                "AD57665422",
                16,
            ),
            x_qA=int(
                "0A420517E406AAC0ACDCE90FCD71487718D3B953EFD7FBEC5F7F27E28"
                "C6149999397E91E029E06457DB2D3E640668B392C2A7E737A7F0BF044"
                "36D11640FD09FD",
                16,
            ),
            y_qA=int(
                "72E6882E8DB28AAD36237CD25D580DB23783961C8DC52DFA2EC138AD4"
                "72A0FCEF3887CF62B623B2A87DE5C588301EA3E5FC269B373B60724F5"
                "E82A6AD147FDE7",
                16,
            ),
            dB=int(
                "230E18E1BCC88A362FA54E4EA3902009292F7F8033624FD471B5D8ACE49"
                "D12CFABBC19963DAB8E2F1EBA00BFFB29E4D72D13F2224562F405CB8050"
                "3666B25429",
                16,
            ),
            x_qB=int(
                "9D45F66DE5D67E2E6DB6E93A59CE0BB48106097FF78A081DE781CDB31"
                "FCE8CCBAAEA8DD4320C4119F1E9CD437A2EAB3731FA9668AB268D871D"
                "EDA55A5473199F",
                16,
            ),
            y_qB=int(
                "2FDC313095BCDD5FB3A91636F07A959C8E86B5636A1E930E8396049CB"
                "481961D365CC11453A06C719835475B12CB52FC3C383BCE35E27EF194"
                "512B71876285FA",
                16,
            ),
            x_Z=int(
                "A7927098655F1F9976FA50A9D566865DC530331846381C87256BAF3226"
                "244B76D36403C024D7BBF0AA0803EAFF405D3D24F11A9B5C0BEF679FE1"
                "454B21C4CD1F",
                16,
            ),
            y_Z=int(
                "7DB71C3DEF63212841C463E881BDCF055523BD368240E6C3143BD8DEF8"
                "B3B3223B95E0F53082FF5E412F4222537A43DF1C6D25729DDB51620A83"
                "2BE6A26680A2",
                16,
            ),
        )


# https://tools.ietf.org/html/rfc4754#page-5
@pytest.mark.parametrize(
    "w, gwx, gwy, k, msg, md, r, s, curve",
    [
        pytest.param(
            "DC51D3866A15BACDE33D96F992FCA99DA7E6EF0934E7097559C27F1614C88A7F",
            "2442A5CC0ECD015FA3CA31DC8E2BBC70BF42D60CBCA20085E0822CB04235E970",
            "6FC98BD7E50211A4A27102FA3549DF79EBCB4BF246B80945CDDFE7D509BBFD7D",
            "9E56F509196784D963D1C0A401510EE7ADA3DCC5DEE04B154BF61AF1D5A6DECE",
            b"abc",
            hashlib.sha256,
            "CB28E0999B9C7715FD0A80D8E47A77079716CBBF917DD72E97566EA1C066957C",
            "86FA3BB4E26CAD5BF90B7F81899256CE7594BB1EA0C89212748BFF3B3D5B0315",
            NIST256p,
            id="ECDSA-256",
        ),
        pytest.param(
            "0BEB646634BA87735D77AE4809A0EBEA865535DE4C1E1DCB692E84708E81A5AF"
            "62E528C38B2A81B35309668D73524D9F",
            "96281BF8DD5E0525CA049C048D345D3082968D10FEDF5C5ACA0C64E6465A97EA"
            "5CE10C9DFEC21797415710721F437922",
            "447688BA94708EB6E2E4D59F6AB6D7EDFF9301D249FE49C33096655F5D502FAD"
            "3D383B91C5E7EDAA2B714CC99D5743CA",
            "B4B74E44D71A13D568003D7489908D564C7761E229C58CBFA18950096EB7463B"
            "854D7FA992F934D927376285E63414FA",
            b"abc",
            hashlib.sha384,
            "FB017B914E29149432D8BAC29A514640B46F53DDAB2C69948084E2930F1C8F7E"
            "08E07C9C63F2D21A07DCB56A6AF56EB3",
            "B263A1305E057F984D38726A1B46874109F417BCA112674C528262A40A629AF1"
            "CBB9F516CE0FA7D2FF630863A00E8B9F",
            NIST384p,
            id="ECDSA-384",
        ),
        pytest.param(
            "0065FDA3409451DCAB0A0EAD45495112A3D813C17BFD34BDF8C1209D7DF58491"
            "20597779060A7FF9D704ADF78B570FFAD6F062E95C7E0C5D5481C5B153B48B37"
            "5FA1",
            "0151518F1AF0F563517EDD5485190DF95A4BF57B5CBA4CF2A9A3F6474725A35F"
            "7AFE0A6DDEB8BEDBCD6A197E592D40188901CECD650699C9B5E456AEA5ADD190"
            "52A8",
            "006F3B142EA1BFFF7E2837AD44C9E4FF6D2D34C73184BBAD90026DD5E6E85317"
            "D9DF45CAD7803C6C20035B2F3FF63AFF4E1BA64D1C077577DA3F4286C58F0AEA"
            "E643",
            "00C1C2B305419F5A41344D7E4359933D734096F556197A9B244342B8B62F46F9"
            "373778F9DE6B6497B1EF825FF24F42F9B4A4BD7382CFC3378A540B1B7F0C1B95"
            "6C2F",
            b"abc",
            hashlib.sha512,
            "0154FD3836AF92D0DCA57DD5341D3053988534FDE8318FC6AAAAB68E2E6F4339"
            "B19F2F281A7E0B22C269D93CF8794A9278880ED7DBB8D9362CAEACEE54432055"
            "2251",
            "017705A7030290D1CEB605A9A1BB03FF9CDD521E87A696EC926C8C10C8362DF4"
            "975367101F67D1CF9BCCBF2F3D239534FA509E70AAC851AE01AAC68D62F86647"
            "2660",
            NIST521p,
            id="ECDSA-521",
        ),
    ],
)
def test_RFC4754_vectors(w, gwx, gwy, k, msg, md, r, s, curve):
    sk = SigningKey.from_string(unhexlify(w), curve)
    vk = VerifyingKey.from_string(unhexlify(gwx + gwy), curve)
    assert sk.verifying_key == vk
    sig = sk.sign(msg, hashfunc=md, sigencode=sigencode_strings, k=int(k, 16))

    assert sig == (unhexlify(r), unhexlify(s))

    assert vk.verify(sig, msg, md, sigdecode_strings)


# key pairs from RFC 6979 appendix A (private key, Ux, Uy)
RFC6979_KEYS = {
    "NIST192p": (
        "6FAB034934E4C0FC9AE67F5B5659A9D7D1FEFD187EE09FD4",
        "AC2C77F529F91689FEA0EA5EFEC7F210D8EEA0B9E047ED56",
        "3BC723E57670BD4887EBC732C523063D0A7C957BC97C1C43",
    ),
    "NIST224p": (
        "F220266E1105BFE3083E03EC7A3A654651F45E37167E88600BF257C1",
        "00CF08DA5AD719E42707FA431292DEA11244D64FC51610D94B130D6C",
        "EEAB6F3DEBE455E3DBF85416F7030CBD94F34F2D6F232C69F3C1385A",
    ),
    "NIST256p": (
        "C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721",
        "60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6",
        "7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299",
    ),
    "NIST384p": (
        "6B9D3DAD2E1B8C1C05B19875B6659F4DE23C3B667BF297BA9AA47740787137D896D57"
        "24E4C70A825F872C9EA60D2EDF5",
        "EC3A4E415B4E19A4568618029F427FA5DA9A8BC4AE92E02E06AAE5286B300C64DEF8F"
        "0EA9055866064A254515480BC13",
        "8015D9B72D7D57244EA8EF9AC0C621896708A59367F9DFB9F54CA84B3F1C9DB1288B2"
        "31C3AE0D4FE7344FD2533264720",
    ),
    "NIST521p": (
        "0FAD06DAA62BA3B25D2FB40133DA757205DE67F5BB0018FEE8C86E1B68C7E75CAA896"
        "EB32F1F47C70855836A6D16FCC1466F6D8FBEC67DB89EC0C08B0E996B83538",
        "1894550D0785932E00EAA23B694F213F8C3121F86DC97A04E5A7167DB4E5BCD371123"
        "D46E45DB6B5D5370A7F20FB633155D38FFA16D2BD761DCAC474B9A2F5023A4",
        "0493101C962CD4D2FDDF782285E64584139C2F91B47F87FF82354D6630F746A28A0DB"
        "25741B5B34A828008B22ACC23F924FAAFBD4D33F81EA66956DFEAA2BFDFCF5",
    ),
}


@pytest.mark.parametrize(
    "msg, md, k, r, s, curve",
    [
        # NIST192p (RFC 6979 section A.2.3)
        pytest.param(
            b"sample",
            hashlib.sha1,
            "37D7CA00D2C7B0E5E412AC03BD44BA837FDD5B28CD3B0021",
            "98C6BD12B23EAF5E2A2045132086BE3EB8EBD62ABF6698FF",
            "57A22B07DEA9530F8DE9471B1DC6624472E8E2844BC25B64",
            NIST192p,
            id="NIST192p-SHA-1-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha224,
            "4381526B3FC1E7128F202E194505592F01D5FF4C5AF015D8",
            "A1F00DAD97AEEC91C95585F36200C65F3C01812AA60378F5",
            "E07EC1304C7C6C9DEBBE980B9692668F81D4DE7922A0F97A",
            NIST192p,
            id="NIST192p-SHA-224-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha256,
            "32B1B6D7D42A05CB449065727A84804FB1A3E34D8F261496",
            "4B0B8CE98A92866A2820E20AA6B75B56382E0F9BFD5ECB55",
            "CCDB006926EA9565CBADC840829D8C384E06DE1F1E381B85",
            NIST192p,
            id="NIST192p-SHA-256-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha384,
            "4730005C4FCB01834C063A7B6760096DBE284B8252EF4311",
            "DA63BF0B9ABCF948FBB1E9167F136145F7A20426DCC287D5",
            "C3AA2C960972BD7A2003A57E1C4C77F0578F8AE95E31EC5E",
            NIST192p,
            id="NIST192p-SHA-384-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha512,
            "A2AC7AB055E4F20692D49209544C203A7D1F2C0BFBC75DB1",
            "4D60C5AB1996BD848343B31C00850205E2EA6922DAC2E4B8",
            "3F6E837448F027A1BF4B34E796E32A811CBB4050908D8F67",
            NIST192p,
            id="NIST192p-SHA-512-sample",
        ),
        pytest.param(
            b"test",
            hashlib.sha1,
            "D9CF9C3D3297D3260773A1DA7418DB5537AB8DD93DE7FA25",
            "0F2141A0EBBC44D2E1AF90A50EBCFCE5E197B3B7D4DE036D",
            "EB18BC9E1F3D7387500CB99CF5F7C157070A8961E38700B7",
            NIST192p,
            id="NIST192p-SHA-1-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha224,
            "F5DC805F76EF851800700CCE82E7B98D8911B7D510059FBE",
            "6945A1C1D1B2206B8145548F633BB61CEF04891BAF26ED34",
            "B7FB7FDFC339C0B9BD61A9F5A8EAF9BE58FC5CBA2CB15293",
            NIST192p,
            id="NIST192p-SHA-224-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha256,
            "5C4CE89CF56D9E7C77C8585339B006B97B5F0680B4306C6C",
            "3A718BD8B4926C3B52EE6BBE67EF79B18CB6EB62B1AD97AE",
            "5662E6848A4A19B1F1AE2F72ACD4B8BBE50F1EAC65D9124F",
            NIST192p,
            id="NIST192p-SHA-256-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha384,
            "5AFEFB5D3393261B828DB6C91FBC68C230727B030C975693",
            "B234B60B4DB75A733E19280A7A6034BD6B1EE88AF5332367",
            "7994090B2D59BB782BE57E74A44C9A1C700413F8ABEFE77A",
            NIST192p,
            id="NIST192p-SHA-384-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha512,
            "0758753A5254759C7CFBAD2E2D9B0792EEE44136C9480527",
            "FE4F4AE86A58B6507946715934FE2D8FF9D95B6B098FE739",
            "74CF5605C98FBA0E1EF34D4B5A1577A7DCF59457CAE52290",
            NIST192p,
            id="NIST192p-SHA-512-test",
        ),
        # NIST224p (RFC 6979 section A.2.4)
        pytest.param(
            b"sample",
            hashlib.sha1,
            "7EEFADD91110D8DE6C2C470831387C50D3357F7F4D477054B8B426BC",
            "22226F9D40A96E19C4A301CE5B74B115303C0F3A4FD30FC257FB57AC",
            "66D1CDD83E3AF75605DD6E2FEFF196D30AA7ED7A2EDF7AF475403D69",
            NIST224p,
            id="NIST224p-SHA-1-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha224,
            "C1D1F2F10881088301880506805FEB4825FE09ACB6816C36991AA06D",
            "1CDFE6662DDE1E4A1EC4CDEDF6A1F5A2FB7FBD9145C12113E6ABFD3E",
            "A6694FD7718A21053F225D3F46197CA699D45006C06F871808F43EBC",
            NIST224p,
            id="NIST224p-SHA-224-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha256,
            "AD3029E0278F80643DE33917CE6908C70A8FF50A411F06E41DEDFCDC",
            "61AA3DA010E8E8406C656BC477A7A7189895E7E840CDFE8FF42307BA",
            "BC814050DAB5D23770879494F9E0A680DC1AF7161991BDE692B10101",
            NIST224p,
            id="NIST224p-SHA-256-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha384,
            "52B40F5A9D3D13040F494E83D3906C6079F29981035C7BD51E5CAC40",
            "0B115E5E36F0F9EC81F1325A5952878D745E19D7BB3EABFABA77E953",
            "830F34CCDFE826CCFDC81EB4129772E20E122348A2BBD889A1B1AF1D",
            NIST224p,
            id="NIST224p-SHA-384-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha512,
            "9DB103FFEDEDF9CFDBA05184F925400C1653B8501BAB89CEA0FBEC14",
            "074BD1D979D5F32BF958DDC61E4FB4872ADCAFEB2256497CDAC30397",
            "A4CECA196C3D5A1FF31027B33185DC8EE43F288B21AB342E5D8EB084",
            NIST224p,
            id="NIST224p-SHA-512-sample",
        ),
        pytest.param(
            b"test",
            hashlib.sha1,
            "2519178F82C3F0E4F87ED5883A4E114E5B7A6E374043D8EFD329C253",
            "DEAA646EC2AF2EA8AD53ED66B2E2DDAA49A12EFD8356561451F3E21C",
            "95987796F6CF2062AB8135271DE56AE55366C045F6D9593F53787BD2",
            NIST224p,
            id="NIST224p-SHA-1-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha224,
            "DF8B38D40DCA3E077D0AC520BF56B6D565134D9B5F2EAE0D34900524",
            "C441CE8E261DED634E4CF84910E4C5D1D22C5CF3B732BB204DBEF019",
            "902F42847A63BDC5F6046ADA114953120F99442D76510150F372A3F4",
            NIST224p,
            id="NIST224p-SHA-224-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha256,
            "FF86F57924DA248D6E44E8154EB69F0AE2AEBAEE9931D0B5A969F904",
            "AD04DDE87B84747A243A631EA47A1BA6D1FAA059149AD2440DE6FBA6",
            "178D49B1AE90E3D8B629BE3DB5683915F4E8C99FDF6E666CF37ADCFD",
            NIST224p,
            id="NIST224p-SHA-256-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha384,
            "7046742B839478C1B5BD31DB2E862AD868E1A45C863585B5F22BDC2D",
            "389B92682E399B26518A95506B52C03BC9379A9DADF3391A21FB0EA4",
            "414A718ED3249FF6DBC5B50C27F71F01F070944DA22AB1F78F559AAB",
            NIST224p,
            id="NIST224p-SHA-384-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha512,
            "E39C2AA4EA6BE2306C72126D40ED77BF9739BB4D6EF2BBB1DCB6169D",
            "049F050477C5ADD858CAC56208394B5A55BAEBBE887FDF765047C17C",
            "077EB13E7005929CEFA3CD0403C7CDCC077ADF4E44F3C41B2F60ECFF",
            NIST224p,
            id="NIST224p-SHA-512-test",
        ),
        # NIST256p (RFC 6979 section A.2.5)
        pytest.param(
            b"sample",
            hashlib.sha1,
            "882905F1227FD620FBF2ABF21244F0BA83D0DC3A9103DBBEE43A1FB858109DB4",
            "61340C88C3AAEBEB4F6D667F672CA9759A6CCAA9FA8811313039EE4A35471D32",
            "6D7F147DAC089441BB2E2FE8F7A3FA264B9C475098FDCF6E00D7C996E1B8B7EB",
            NIST256p,
            id="NIST256p-SHA-1-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha224,
            "103F90EE9DC52E5E7FB5132B7033C63066D194321491862059967C715985D473",
            "53B2FFF5D1752B2C689DF257C04C40A587FABABB3F6FC2702F1343AF7CA9AA3F",
            "B9AFB64FDC03DC1A131C7D2386D11E349F070AA432A4ACC918BEA988BF75C74C",
            NIST256p,
            id="NIST256p-SHA-224-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha256,
            "A6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60",
            "EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716",
            "F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8",
            NIST256p,
            id="NIST256p-SHA-256-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha384,
            "09F634B188CEFD98E7EC88B1AA9852D734D0BC272F7D2A47DECC6EBEB375AAD4",
            "0EAFEA039B20E9B42309FB1D89E213057CBF973DC0CFC8F129EDDDC800EF7719",
            "4861F0491E6998B9455193E34E7B0D284DDD7149A74B95B9261F13ABDE940954",
            NIST256p,
            id="NIST256p-SHA-384-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha512,
            "5FA81C63109BADB88C1F367B47DA606DA28CAD69AA22C4FE6AD7DF73A7173AA5",
            "8496A60B5E9B47C825488827E0495B0E3FA109EC4568FD3F8D1097678EB97F00",
            "2362AB1ADBE2B8ADF9CB9EDAB740EA6049C028114F2460F96554F61FAE3302FE",
            NIST256p,
            id="NIST256p-SHA-512-sample",
        ),
        pytest.param(
            b"test",
            hashlib.sha1,
            "8C9520267C55D6B980DF741E56B4ADEE114D84FBFA2E62137954164028632A2E",
            "0CBCC86FD6ABD1D99E703E1EC50069EE5C0B4BA4B9AC60E409E8EC5910D81A89",
            "01B9D7B73DFAA60D5651EC4591A0136F87653E0FD780C3B1BC872FFDEAE479B1",
            NIST256p,
            id="NIST256p-SHA-1-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha224,
            "669F4426F2688B8BE0DB3A6BD1989BDAEFFF84B649EEB84F3DD26080F667FAA7",
            "C37EDB6F0AE79D47C3C27E962FA269BB4F441770357E114EE511F662EC34A692",
            "C820053A05791E521FCAAD6042D40AEA1D6B1A540138558F47D0719800E18F2D",
            NIST256p,
            id="NIST256p-SHA-224-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha256,
            "D16B6AE827F17175E040871A1C7EC3500192C4C92677336EC2537ACAEE0008E0",
            "F1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367",
            "019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083",
            NIST256p,
            id="NIST256p-SHA-256-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha384,
            "16AEFFA357260B04B1DD199693960740066C1A8F3E8EDD79070AA914D361B3B8",
            "83910E8B48BB0C74244EBDF7F07A1C5413D61472BD941EF3920E623FBCCEBEB6",
            "8DDBEC54CF8CD5874883841D712142A56A8D0F218F5003CB0296B6B509619F2C",
            NIST256p,
            id="NIST256p-SHA-384-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha512,
            "6915D11632ACA3C40D5D51C08DAF9C555933819548784480E93499000D9F0B7F",
            "461D93F31B6540894788FD206C07CFA0CC35F46FA3C91816FFF1040AD1581A04",
            "39AF9F15DE0DB8D97E72719C74820D304CE5226E32DEDAE67519E840D1194E55",
            NIST256p,
            id="NIST256p-SHA-512-test",
        ),
        # NIST384p (RFC 6979 section A.2.6)
        pytest.param(
            b"sample",
            hashlib.sha1,
            "4471EF7518BB2C7C20F62EAE1C387AD0C5E8E470995DB4ACF694466E6AB096630"
            "F29E5938D25106C3C340045A2DB01A7",
            "EC748D839243D6FBEF4FC5C4859A7DFFD7F3ABDDF72014540C16D73309834FA37"
            "B9BA002899F6FDA3A4A9386790D4EB2",
            "A3BCFA947BEEF4732BF247AC17F71676CB31A847B9FF0CBC9C9ED4C1A5B3FACF2"
            "6F49CA031D4857570CCB5CA4424A443",
            NIST384p,
            id="NIST384p-SHA-1-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha224,
            "A4E4D2F0E729EB786B31FC20AD5D849E304450E0AE8E3E341134A5C1AFA03CAB8"
            "083EE4E3C45B06A5899EA56C51B5879",
            "42356E76B55A6D9B4631C865445DBE54E056D3B3431766D0509244793C3F93664"
            "50F76EE3DE43F5A125333A6BE060122",
            "9DA0C81787064021E78DF658F2FBB0B042BF304665DB721F077A4298B095E4834"
            "C082C03D83028EFBF93A3C23940CA8D",
            NIST384p,
            id="NIST384p-SHA-224-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha256,
            "180AE9F9AEC5438A44BC159A1FCB277C7BE54FA20E7CF404B490650A8ACC414E3"
            "75572342863C899F9F2EDF9747A9B60",
            "21B13D1E013C7FA1392D03C5F99AF8B30C570C6F98D4EA8E354B63A21D3DAA33B"
            "DE1E888E63355D92FA2B3C36D8FB2CD",
            "F3AA443FB107745BF4BD77CB3891674632068A10CA67E3D45DB2266FA7D1FEEBE"
            "FDC63ECCD1AC42EC0CB8668A4FA0AB0",
            NIST384p,
            id="NIST384p-SHA-256-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha384,
            "94ED910D1A099DAD3254E9242AE85ABDE4BA15168EAF0CA87A555FD56D10FBCA2"
            "907E3E83BA95368623B8C4686915CF9",
            "94EDBB92A5ECB8AAD4736E56C691916B3F88140666CE9FA73D64C4EA95AD133C8"
            "1A648152E44ACF96E36DD1E80FABE46",
            "99EF4AEB15F178CEA1FE40DB2603138F130E740A19624526203B6351D0A3A94FA"
            "329C145786E679E7B82C71A38628AC8",
            NIST384p,
            id="NIST384p-SHA-384-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha512,
            "92FC3C7183A883E24216D1141F1A8976C5B0DD797DFA597E3D7B32198BD35331A"
            "4E966532593A52980D0E3AAA5E10EC3",
            "ED0959D5880AB2D869AE7F6C2915C6D60F96507F9CB3E047C0046861DA4A799CF"
            "E30F35CC900056D7C99CD7882433709",
            "512C8CCEEE3890A84058CE1E22DBC2198F42323CE8ACA9135329F03C068E5112D"
            "C7CC3EF3446DEFCEB01A45C2667FDD5",
            NIST384p,
            id="NIST384p-SHA-512-sample",
        ),
        pytest.param(
            b"test",
            hashlib.sha1,
            "66CC2C8F4D303FC962E5FF6A27BD79F84EC812DDAE58CF5243B64A4AD8094D47E"
            "C3727F3A3C186C15054492E30698497",
            "4BC35D3A50EF4E30576F58CD96CE6BF638025EE624004A1F7789A8B8E43D0678A"
            "CD9D29876DAF46638645F7F404B11C7",
            "D5A6326C494ED3FF614703878961C0FDE7B2C278F9A65FD8C4B7186201A299169"
            "5BA1C84541327E966FA7B50F7382282",
            NIST384p,
            id="NIST384p-SHA-1-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha224,
            "18FA39DB95AA5F561F30FA3591DC59C0FA3653A80DAFFA0B48D1A4C6DFCBFF6E3"
            "D33BE4DC5EB8886A8ECD093F2935726",
            "E8C9D0B6EA72A0E7837FEA1D14A1A9557F29FAA45D3E7EE888FC5BF954B5E6246"
            "4A9A817C47FF78B8C11066B24080E72",
            "07041D4A7A0379AC7232FF72E6F77B6DDB8F09B16CCE0EC3286B2BD43FA8C6141"
            "C53EA5ABEF0D8231077A04540A96B66",
            NIST384p,
            id="NIST384p-SHA-224-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha256,
            "0CFAC37587532347DC3389FDC98286BBA8C73807285B184C83E62E26C401C0FAA"
            "48DD070BA79921A3457ABFF2D630AD7",
            "6D6DEFAC9AB64DABAFE36C6BF510352A4CC27001263638E5B16D9BB51D451559F"
            "918EEDAF2293BE5B475CC8F0188636B",
            "2D46F3BECBCC523D5F1A1256BF0C9B024D879BA9E838144C8BA6BAEB4B53B47D5"
            "1AB373F9845C0514EEFB14024787265",
            NIST384p,
            id="NIST384p-SHA-256-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha384,
            "015EE46A5BF88773ED9123A5AB0807962D193719503C527B031B4C2D225092ADA"
            "71F4A459BC0DA98ADB95837DB8312EA",
            "8203B63D3C853E8D77227FB377BCF7B7B772E97892A80F36AB775D509D7A5FEB0"
            "542A7F0812998DA8F1DD3CA3CF023DB",
            "DDD0760448D42D8A43AF45AF836FCE4DE8BE06B485E9B61B827C2F13173923E06"
            "A739F040649A667BF3B828246BAA5A5",
            NIST384p,
            id="NIST384p-SHA-384-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha512,
            "3780C4F67CB15518B6ACAE34C9F83568D2E12E47DEAB6C50A4E4EE5319D1E8CE0"
            "E2CC8A136036DC4B9C00E6888F66B6C",
            "A0D5D090C9980FAF3C2CE57B7AE951D31977DD11C775D314AF55F76C676447D06"
            "FB6495CD21B4B6E340FC236584FB277",
            "976984E59B4C77B0E8E4460DCA3D9F20E07B9BB1F63BEEFAF576F6B2E8B224634"
            "A2092CD3792E0159AD9CEE37659C736",
            NIST384p,
            id="NIST384p-SHA-512-test",
        ),
        # NIST521p (RFC 6979 section A.2.7)
        pytest.param(
            b"sample",
            hashlib.sha1,
            "089C071B419E1C2820962321787258469511958E80582E95D8378E0C2CCDB3CB4"
            "2BEDE42F50E3FA3C71F5A76724281D31D9C89F0F91FC1BE4918DB1C03A5838D0F"
            "9",
            "0343B6EC45728975EA5CBA6659BBB6062A5FF89EEA58BE3C80B619F322C87910F"
            "E092F7D45BB0F8EEE01ED3F20BABEC079D202AE677B243AB40B5431D497C55D75"
            "D",
            "0E7B0E675A9B24413D448B8CC119D2BF7B2D2DF032741C096634D6D65D0DBE3D5"
            "694625FB9E8104D3B842C1B0E2D0B98BEA19341E8676AEF66AE4EBA3D5475D5D1"
            "6",
            NIST521p,
            id="NIST521p-SHA-1-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha224,
            "121415EC2CD7726330A61F7F3FA5DE14BE9436019C4DB8CB4041F3B54CF31BE04"
            "93EE3F427FB906393D895A19C9523F3A1D54BB8702BD4AA9C99DAB2597B92113F"
            "3",
            "1776331CFCDF927D666E032E00CF776187BC9FDD8E69D0DABB4109FFE1B5E2A30"
            "715F4CC923A4A5E94D2503E9ACFED92857B7F31D7152E0F8C00C15FF3D87E2ED2"
            "E",
            "050CB5265417FE2320BBB5A122B8E1A32BD699089851128E360E620A30C7E17BA"
            "41A666AF126CE100E5799B153B60528D5300D08489CA9178FB610A2006C254B41"
            "F",
            NIST521p,
            id="NIST521p-SHA-224-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha256,
            "0EDF38AFCAAECAB4383358B34D67C9F2216C8382AAEA44A3DAD5FDC9C32575761"
            "793FEF24EB0FC276DFC4F6E3EC476752F043CF01415387470BCBD8678ED2C7E1A"
            "0",
            "1511BB4D675114FE266FC4372B87682BAECC01D3CC62CF2303C92B3526012659D"
            "16876E25C7C1E57648F23B73564D67F61C6F14D527D54972810421E7D87589E1A"
            "7",
            "04A171143A83163D6DF460AAF61522695F207A58B95C0644D87E52AA1A347916E"
            "4F7A72930B1BC06DBE22CE3F58264AFD23704CBB63B29B931F7DE6C9D949A7ECF"
            "C",
            NIST521p,
            id="NIST521p-SHA-256-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha384,
            "1546A108BC23A15D6F21872F7DED661FA8431DDBD922D0DCDB77CC878C8553FFA"
            "D064C95A920A750AC9137E527390D2D92F153E66196966EA554D9ADFCB109C421"
            "1",
            "1EA842A0E17D2DE4F92C15315C63DDF72685C18195C2BB95E572B9C5136CA4B4B"
            "576AD712A52BE9730627D16054BA40CC0B8D3FF035B12AE75168397F5D50C6745"
            "1",
            "1F21A3CEE066E1961025FB048BD5FE2B7924D0CD797BABE0A83B66F1E35EEAF5F"
            "DE143FA85DC394A7DEE766523393784484BDF3E00114A1C857CDE1AA203DB65D6"
            "1",
            NIST521p,
            id="NIST521p-SHA-384-sample",
        ),
        pytest.param(
            b"sample",
            hashlib.sha512,
            "1DAE2EA071F8110DC26882D4D5EAE0621A3256FC8847FB9022E2B7D28E6F10198"
            "B1574FDD03A9053C08A1854A168AA5A57470EC97DD5CE090124EF52A2F7ECBFFD"
            "3",
            "0C328FAFCBD79DD77850370C46325D987CB525569FB63C5D3BC53950E6D4C5F17"
            "4E25A1EE9017B5D450606ADD152B534931D7D4E8455CC91F9B15BF05EC36E377F"
            "A",
            "0617CCE7CF5064806C467F678D3B4080D6F1CC50AF26CA209417308281B68AF28"
            "2623EAA63E5B5C0723D8B8C37FF0777B1A20F8CCB1DCCC43997F1EE0E44DA4A67"
            "A",
            NIST521p,
            id="NIST521p-SHA-512-sample",
        ),
        pytest.param(
            b"test",
            hashlib.sha1,
            "0BB9F2BF4FE1038CCF4DABD7139A56F6FD8BB1386561BD3C6A4FC818B20DF5DDB"
            "A80795A947107A1AB9D12DAA615B1ADE4F7A9DC05E8E6311150F47F5C57CE8B22"
            "2",
            "13BAD9F29ABE20DE37EBEB823C252CA0F63361284015A3BF430A46AAA80B87B06"
            "93F0694BD88AFE4E661FC33B094CD3B7963BED5A727ED8BD6A3A202ABE009D036"
            "7",
            "1E9BB81FF7944CA409AD138DBBEE228E1AFCC0C890FC78EC8604639CB0DBDC90F"
            "717A99EAD9D272855D00162EE9527567DD6A92CBD629805C0445282BBC916797F"
            "F",
            NIST521p,
            id="NIST521p-SHA-1-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha224,
            "040D09FCF3C8A5F62CF4FB223CBBB2B9937F6B0577C27020A99602C25A0113698"
            "7E452988781484EDBBCF1C47E554E7FC901BC3085E5206D9F619CFF07E73D6F70"
            "6",
            "1C7ED902E123E6815546065A2C4AF977B22AA8EADDB68B2C1110E7EA44D42086B"
            "FE4A34B67DDC0E17E96536E358219B23A706C6A6E16BA77B65E1C595D43CAE17F"
            "B",
            "177336676304FCB343CE028B38E7B4FBA76C1C1B277DA18CAD2A8478B2A9A9F5B"
            "EC0F3BA04F35DB3E4263569EC6AADE8C92746E4C82F8299AE1B8F1739F8FD519A"
            "4",
            NIST521p,
            id="NIST521p-SHA-224-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha256,
            "01DE74955EFAABC4C4F17F8E84D881D1310B5392D7700275F82F145C61E843841"
            "AF09035BF7A6210F5A431A6A9E81C9323354A9E69135D44EBD2FCAA7731B90925"
            "8",
            "00E871C4A14F993C6C7369501900C4BC1E9C7B0B4BA44E04868B30B41D8071042"
            "EB28C4C250411D0CE08CD197E4188EA4876F279F90B3D8D74A3C76E6F1E4656AA"
            "8",
            "0CD52DBAA33B063C3A6CD8058A1FB0A46A4754B034FCC644766CA14DA8CA5CA9F"
            "DE00E88C1AD60CCBA759025299079D7A427EC3CC5B619BFBC828E7769BCD694E8"
            "6",
            NIST521p,
            id="NIST521p-SHA-256-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha384,
            "1F1FC4A349A7DA9A9E116BFDD055DC08E78252FF8E23AC276AC88B1770AE0B5DC"
            "EB1ED14A4916B769A523CE1E90BA22846AF11DF8B300C38818F713DADD85DE0C8"
            "8",
            "14BEE21A18B6D8B3C93FAB08D43E739707953244FDBE924FA926D76669E7AC8C8"
            "9DF62ED8975C2D8397A65A49DCC09F6B0AC62272741924D479354D74FF6075578"
            "C",
            "133330865C067A0EAF72362A65E2D7BC4E461E8C8995C3B6226A21BD1AA78F0ED"
            "94FE536A0DCA35534F0CD1510C41525D163FE9D74D134881E35141ED5E8E95B97"
            "9",
            NIST521p,
            id="NIST521p-SHA-384-test",
        ),
        pytest.param(
            b"test",
            hashlib.sha512,
            "16200813020EC986863BEDFC1B121F605C1215645018AEA1A7B215A564DE9EB1B"
            "38A67AA1128B80CE391C4FB71187654AAA3431027BFC7F395766CA988C964DC56"
            "D",
            "13E99020ABF5CEE7525D16B69B229652AB6BDF2AFFCAEF38773B4B7D08725F10C"
            "DB93482FDCC54EDCEE91ECA4166B2A7C6265EF0CE2BD7051B7CEF945BABD47EE6"
            "D",
            "1FBD0013C674AA79CB39849527916CE301C66EA7CE8B80682786AD60F98F7E78A"
            "19CA69EFF5C57400E3B3A0AD66CE0978214D13BAF4E9AC60752F7B155E2DE4DCE"
            "3",
            NIST521p,
            id="NIST521p-SHA-512-test",
        ),
    ],
)
def test_RFC6979_signature_vectors(msg, md, k, r, s, curve):
    """
    End-to-end signature vectors from RFC 6979, appendix A.2.3 to A.2.7.

    Verifies the whole deterministic signing process against the
    reference (r, s) values, not just the nonce generation.
    """
    x, ux, uy = RFC6979_KEYS[curve.name]
    sk = SigningKey.from_secret_exponent(int(x, 16), curve=curve, hashfunc=md)
    vk = sk.verifying_key

    # the public key derived from the private key must match the key
    # pair published in the RFC
    assert vk.pubkey.point.x() == int(ux, 16)
    assert vk.pubkey.point.y() == int(uy, 16)

    # the nonce generated by the deterministic process must match too
    digest = md(msg).digest()
    gen_k = rfc6979.generate_k(curve.order, int(x, 16), md, digest)
    assert gen_k == int(k, 16)

    sig = sk.sign_deterministic(msg, hashfunc=md, sigencode=sigencode_strings)
    assert sig == (
        number_to_string(int(r, 16), curve.order),
        number_to_string(int(s, 16), curve.order),
    )

    assert vk.verify(sig, msg, md, sigdecode_strings, allow_truncate=True)

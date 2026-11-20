"""Synthetic privacy test set (data/privacy_test.jsonl). All values are fake / Luhn-generated test numbers."""
import json

S, D = [], []
def s(kind, text, *sens): S.append(dict(kind=kind, text=text, sensitive=list(sens), decoy=False))
def d(text): D.append(dict(kind="decoy", text=text, sensitive=[], decoy=True))

# OTP (English / Hinglish / Hindi, varied wording)
for t, v in [("Your OTP is 482913. Do not share it.", "482913"), ("OTP: 7391", "7391"), ("Use verification code 558201 to log in", "558201"),
             ("482913 is your OTP for the transaction", "482913"), ("One-time password 90817265 valid for 5 minutes", "90817265"),
             ("Aapka OTP 665412 hai, kisi se share na karein", "665412"), ("आपका OTP 301982 है", "301982"),
             ("Your authentication code is 224466", "224466"), ("security code: 8821", "8821"), ("login code 119203 expires soon", "119203"),
             ("OTP for payment of Rs 500 is 774411", "774411"), ("The otp I received was 3344 but it failed", "3344")]:
    s("otp", t, v)
# cards (Luhn-valid test numbers)
for t, v in [("My card number is 4111 1111 1111 1111", "4111 1111 1111 1111"), ("pay with 5500-0000-0000-0004", "5500-0000-0000-0004"),
             ("card 378282246310005 expiry 12/28", "378282246310005"), ("4012888888881881 is the visa", "4012888888881881"),
             ("Use this debit card: 6011111111111117", "6011111111111117"), ("card no 3530 1113 3330 0000", "3530 1113 3330 0000")]:
    s("card", t, v)
s("cvv", "cvv 123 and expiry 09/27", "123"); s("cvv", "CVV: 4821", "4821")
s("pin", "My ATM PIN is 4821", "4821"); s("pin", "UPI pin 112233 do not tell anyone", "112233")
# IDs / banking
s("aadhaar", "Aadhaar 2345 6789 0123", "2345 6789 0123"); s("aadhaar", "my aadhaar number 234567890123", "234567890123"); s("aadhaar", "आधार 345678901234", "345678901234")
s("pan", "PAN is ABCDE1234F", "ABCDE1234F"); s("pan", "pan card FGHIJ5678K attached", "FGHIJ5678K")
s("upi", "send to rahul.sharma@oksbi", "rahul.sharma@oksbi"); s("upi", "my upi priya123@ybl", "priya123@ybl"); s("upi", "pay anil@paytm now", "anil@paytm")
s("bank_account", "Account number 123456789012", "123456789012"); s("bank_account", "a/c no. 50100234567891", "50100234567891"); s("bank_account", "acc no 00112233445", "00112233445")
s("ifsc", "IFSC HDFC0001234", "HDFC0001234"); s("ifsc", "ifsc code SBIN0004567 branch Pune", "SBIN0004567")
# secrets
s("api_key", "my key is sk-abc123def456ghi789jkl012", "sk-abc123def456ghi789jkl012")
s("api_key", "AWS key AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE")
s("api_key", "token ghp_abcdefghijklmnopqrstuvwxyz0123456789", "ghp_abcdefghijklmnopqrstuvwxyz0123456789")
s("api_key", "slack xoxb-1234567890-abcdefghij", "xoxb-1234567890-abcdefghij")
s("api_key", "AIzaSyA1234567890abcdefghijklmnopqrstuv is the maps key", "AIzaSyA1234567890abcdefghijklmnopqrstuv")
s("jwt", "auth eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.abc123def456ghi", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.abc123def456ghi")
s("token", "Authorization: Bearer 9f8e7d6c5b4a39281716ab", "9f8e7d6c5b4a39281716ab")
s("secret", "API_KEY=supersecretvalue99", "supersecretvalue99"); s("secret", "token: Zx81QpLm20Vb", "Zx81QpLm20Vb")
s("password", "my password is Rahul@123", "Rahul@123"); s("password", "password: Tr0ub4dor&3", "Tr0ub4dor&3")
s("password", "mera password hai Sunshine#77", "Sunshine#77"); s("password", "पासवर्ड है Hello@2026", "Hello@2026")
s("password", "wifi pwd = Home-Net-5588", "Home-Net-5588")

# decoys (must NOT change)
for t in ["Order #48213 has shipped.", "Flight AI 202 departs at 10:30.", "The meeting is on 15/11/2026 in room 304.",
          "My PIN code is 411057 (Pune).", "Total is Rs. 4500 including tax.", "Call me on 98765 43210.",
          "Zoom meeting ID 123 456 7890", "Invoice 20261115 is pending.", "See page 1234 of the report.", "Version 1.2.3 released in 2026.",
          "Don't share your OTP with anyone.", "I did not get the OTP yet.", "The code review is at 3 PM.", "Postal code 400001 Mumbai.",
          "Reminder: call Rahul tomorrow at 10 AM.", "Thursday works for me.", "Kal 11 baje meeting hai.", "Ticket 99120034 booked for Friday.",
          "He scored 4111 points in the game.", "Our address is Flat 12, Kasarwadi, Pune 411034.", "Seat numbers 12 13 14 are reserved.",
          "Temperature will be 32 degrees on 2026-11-16.", "My employee ID is E-20451.", "Transaction history for last 30 days attached.",
          "The password policy requires 12 characters.", "I forgot my password again, so annoying.", "Keep your card safe when travelling.",
          "Tracking number 1Z999AA10123456784 on its way.", "Batch 2026 11 15 08 30 completed.", "Aadhaar linking deadline is next month."]:
    d(t)

with open("data/privacy_test.jsonl", "w", encoding="utf-8") as f:
    for i, it in enumerate(S + D, 1):
        it["id"] = f"P{i:03d}"
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(S), "sensitive,", len(D), "decoys")

# ---- harder, adversarial set: expected to expose gaps in the regex baseline (reported separately)
H = []
def h(kind, text, *sens): H.append(dict(kind=kind, text=text, sensitive=list(sens), decoy=False))
h("otp_spaced", "your otp is 4 8 2 9 1 3", "4 8 2 9 1 3")
h("otp_words", "the otp is four eight two nine one three", "four eight two nine one three")
h("otp_no_keyword", "Use 553920 to confirm the payment, valid 10 min", "553920")
h("card_split", "my card is 4111 1111 and then 1111 1111", "4111 1111", "1111 1111")
h("card_obfuscated", "card: 4111-1111-1111-1111 (dont tell)", "4111-1111-1111-1111")
h("password_no_keyword", "wifi login is Rahul@123 for the guest network", "Rahul@123")
h("secret_no_keyword", "here you go: 9f8e7d6c5b4a39281716ab0c1d2e3f4a", "9f8e7d6c5b4a39281716ab0c1d2e3f4a")
h("password_hinglish_roman", "mera pass Sunshine#77 hai", "Sunshine#77")
h("aadhaar_dashes", "aadhaar 2345-6789-0123", "2345-6789-0123")
h("upi_unknown_bank", "send money to rahul@okkotak", "rahul@okkotak")
h("api_key_in_url", "https://api.example.com/v1?api_key=abcd1234efgh5678", "abcd1234efgh5678")
h("pin_words", "my atm pin is one two three four", "one two three four")
with open("data/privacy_hard.jsonl", "w", encoding="utf-8") as f:
    for i, it in enumerate(H, 1):
        it["id"] = f"H{i:02d}"
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(H), "hard cases")

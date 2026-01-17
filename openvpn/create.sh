#!/bin/bash

cd /usr/local/openvpn_as/scripts/

# Erstelle/Überprüfe Nutzer existiert, lokales auth und Passwort
./sacli --user "ssh-user" --key "user_auth_type" --value "local" UserPropPut
./sacli --user "ssh-user" --new_pass "1234" SetLocalPassword

# Nutzer muss sich verbinden dürfen
./sacli --user "ssh-user" --key "type" --value "user_connect" UserPropPut

# Autologin für diesen Nutzer einstellen
./sacli --user "ssh-user" --key "prop_autologin" --value "true" UserPropPut

# Änderungen speichern
./sacli start

# Das Profil generieren, mit Autologin und im Volume-Directory speichern
./sacli --user "ssh-user" GetAutologin > /profiles/ssh-user-autologin.ovpn

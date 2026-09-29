#!/bin/sh

set -e -u

brew_prefix=$(brew --prefix)
openssl_link="${brew_prefix}/bin/openssl"

# Runner images can retain this link after the old formula has been removed,
# so `brew list` and `brew unlink` cannot reliably clean it up.
if test -L "${openssl_link}"; then
    case "$(readlink "${openssl_link}")" in
        "${brew_prefix}/opt/openssl@1.1/bin/openssl"|../opt/openssl@1.1/bin/openssl)
            rm "${openssl_link}"
            ;;
    esac
fi

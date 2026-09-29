#!/bin/sh

set -e -u

cmake_version=$1
cmake_arch=$2
installer_path=$3
installer_url="https://github.com/Kitware/CMake/releases/download/v${cmake_version}/cmake-${cmake_version}-linux-${cmake_arch}.sh"

for attempt in 1 2 3; do
    if curl --fail --location --retry 3 --connect-timeout 20 --max-time 300 \
        --output "${installer_path}" "${installer_url}"; then
        # A proxy or release endpoint can return an HTML page with HTTP 200.
        # Never execute it as an installer; retry the download instead.
        if head -n 1 "${installer_path}" | grep -qx '#!/bin/sh'; then
            exit 0
        fi
    fi
    echo "CMake installer download attempt ${attempt} was invalid" >&2
    sleep 5
done

echo "Could not download a valid CMake installer" >&2
exit 1

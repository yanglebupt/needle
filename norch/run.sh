cd build
rm -r ./*
cmake .. -G "Unix Makefiles"
# cmake --build . --config Release
make -j4
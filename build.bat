
@echo on
setlocal


if not exist build\ (
	mkdir build
)
cd build

@REM Note on windows we need to use single-config generator (eg. specity -DCMAKE_BUILD_TYPE during config stage)
@REM because the sdk need to link different libraries for debug/release build.

@REM build for debug dll
cmake -A x64 .. -DCMAKE_BUILD_TYPE=Debug  -DFFX_API_BACKEND=vk_windows_x64 -DFFX_BUILD_AS_DLL=ON -DFFX_FSR3_AS_LIBRARY=OFF %*%
cmake --build ./ --verbose --config Debug --parallel 4 -- /p:CL_MPcount=16

@REM build for release dll
cmake -A x64 .. -DCMAKE_BUILD_TYPE=Release  -DFFX_API_BACKEND=vk_windows_x64 -DFFX_BUILD_AS_DLL=ON -DFFX_FSR3_AS_LIBRARY=OFF %*%
cmake --build ./ --verbose --config Release --parallel 4 -- /p:CL_MPcount=16

:: Come back to root level
cd ..
endlocal

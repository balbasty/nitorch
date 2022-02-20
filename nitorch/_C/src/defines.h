#pragma once


#ifndef NI_MAX_NUM_CHANNELS
#  define NI_MAX_NUM_CHANNELS 128
#endif

#define CALL_MEMBER_FN(object,ptrToMember) ((object).*(ptrToMember))

#define MIN(a,b) (a < b ? a : b)
#define MAX(a,b) (a > b ? a : b)

#define CAT(a, ...) PRIMITIVE_CAT(a, __VA_ARGS__)
#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__
#define IIF(c) PRIMITIVE_CAT(IIF_, c)
#define IIF_0(t, ...) __VA_ARGS__
#define IIF_1(t, ...) t

#define DEC(x) PRIMITIVE_CAT(DEC_, x)
#define DEC_1 0
#define DEC_2 1
#define DEC_3 2
#define DEC_4 3
#define DEC_5 4
#define DEC_6 5
#define DEC_7 6
#define DEC_8 7
#define DEC_9 8

#ifdef NI_DEBUG
#  include <cstdio>
#  define NI_TRACE(...) printf(__VA_ARGS__)
#else 
#  define NI_TRACE(...) {}
#endif

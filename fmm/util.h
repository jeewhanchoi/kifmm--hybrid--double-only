#if !defined (INC_UTIL_H)
#define INC_UTIL_H

#if defined (__cplusplus)
extern "C" {
#endif

  int getenv__int (const char* var, int default_value);
  int getenv__flag (const char* var, int default_value);
  int getenv__numa (void);
  int getenv__upbs (void);
  int getenv__accuracy (void);
  int getenv__block_size (void);
  int getenv__validate (void);
  int getenv__coincide (void);

#if defined (__cplusplus)
}
#endif

#endif

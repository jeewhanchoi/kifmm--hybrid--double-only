#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <papi.h>

static
void
enum_events (int mask)
{
  int i = mask;
  PAPI_event_info_t info;

  fprintf (stdout, "Name                        Code        Description\n");
  PAPI_enum_event (&i, PAPI_ENUM_FIRST);
  do {
    int retval = PAPI_get_event_info (i, &info);
    if (retval == PAPI_OK) {
      fprintf (stdout, "%-30s 0x%-10x%s\n", info.symbol, info.event_code, info.long_descr);
    }
  } while (PAPI_enum_event (&i, 0) == PAPI_OK);
  fprintf (stdout, "=== end of event list ===\n");
}

int
main (int argc, char* argv[])
{
  int retval;
  int i; /* iterates over events */

  /* Initialize PAPI library */
  retval = PAPI_library_init (PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT) {
    fprintf (stderr, "*** Error: Couldn't initialize PAPI library. ***\n");
    assert (retval == PAPI_VER_CURRENT);
    exit (1);
  }

  /* Enumerate all available events */
  printf ("==========================================\n"
	  "Preset events\n"
	  "==========================================\n");
  enum_events (PAPI_PRESET_MASK);

  printf ("==========================================\n"
	  "Native events\n"
	  "==========================================\n");
  enum_events (PAPI_NATIVE_MASK);

  return 0;
}

/* eof */

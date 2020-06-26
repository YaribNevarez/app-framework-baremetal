#include "test_app.h"

int main (void)
{
  Result rc;

  TestApp * app = TestApp_instance ();

  rc = (app != NULL) ? OK : ERROR;

  if (rc == OK)
  {
    rc = app->initialize (app);

    if (rc == OK)
    {
      rc = app->run (app);
    }

    app->dispose (app);
  }

  return rc;
}

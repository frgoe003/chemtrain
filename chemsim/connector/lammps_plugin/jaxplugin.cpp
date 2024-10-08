
#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "jax_connect.h"

using namespace LAMMPS_NS;

static Pair *jaxNNCreator(LAMMPS *lmp)
{
  return new JaxConnect(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  // register plain morse2 pair style
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "jaxnn";
  plugin.info = "Potential in JAX";
  plugin.author = "Paul Fuchs (paul.fuchs@tum.de)";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &jaxNNCreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}

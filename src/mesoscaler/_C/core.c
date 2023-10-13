#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "lib.h"

void print_python_version() {
  //
  printf("Python version: %s", Py_GetVersion());
}

static PyObject *py_print_python_version(PyObject *self, PyObject *args) {
  print_python_version();
  return Py_None;
};

static PyObject *py_is_prime(PyObject *self, PyObject *args) {
  //
  int n;
  if (!PyArg_ParseTuple(args, "i", &n)) {
    return NULL;
  }
  return PyBool_FromLong(is_prime(n));
};

static PyMethodDef lib_methods[] = {
    //
    {"is_prime", py_is_prime, METH_VARARGS, "Check if a number is prime"},
    {NULL, NULL, 0, NULL}
    //
};

static struct PyModuleDef lib_module = {
    //
    PyModuleDef_HEAD_INIT, "_C", "Python API", -1, lib_methods
    //
};

PyMODINIT_FUNC PyInit__C(void) {
  //
  return PyModule_Create(&lib_module);
}
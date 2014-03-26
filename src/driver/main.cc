/*
 *  Copyright (C) 2013  Justin Turney
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.

 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-151 USA.
 */

#if defined(HAVE_MPI)
#include <mpi.h>
#include <tensor/cyclops_tensor.h>
#include <util/world.h>
#endif

#include <boost/python/detail/wrap_python.hpp>
#include <boost/python/module.hpp>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <fstream>

#define PY_TRY(ptr, command)  \
     if(!(ptr = command)){    \
         PyErr_Print();       \
         exit(1);             \
     }

const char* interactive =
  "import readline\n"
  "import code\n"
  "import ambit\n"
  "vars = globals().copy()\n"
  "vars.update(locals())\n"
  "shell = code.InteractiveConsole(vars)\n"
  "shell.interact()\n";

using namespace boost::python;

/// @brief Type that allows for registration of conversions from
///        python iterable types.
struct iterable_converter
{
  /// @note Registers converter from a python interable type to the
  ///       provided type.
  template <typename Container>
  iterable_converter&
  from_python()
  {
    boost::python::converter::registry::push_back(
      &iterable_converter::convertible,
      &iterable_converter::construct<Container>,
      boost::python::type_id<Container>());
    return *this;
  }

  /// @brief Check if PyObject is iterable.
  static void* convertible(PyObject* object)
  {
    return PyObject_GetIter(object) ? object : NULL;
  }

  /// @brief Convert iterable PyObject to C++ container type.
  ///
  /// Container Concept requirements:
  ///
  ///   * Container::value_type is CopyConstructable.
  ///   * Container can be constructed and populated with two iterators.
  ///     I.e. Container(begin, end)
  template <typename Container>
  static void construct(
    PyObject* object,
    boost::python::converter::rvalue_from_python_stage1_data* data)
  {
    namespace python = boost::python;
    // Object is a borrowed reference, so create a handle indicting it is
    // borrowed for proper reference counting.
    python::handle<> handle(python::borrowed(object));

    // Obtain a handle to the memory block that the converter has allocated
    // for the C++ type.
    typedef python::converter::rvalue_from_python_storage<Container>
                                                                 storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    typedef python::stl_input_iterator<typename Container::value_type>
                                                                     iterator;

    // Allocate the C++ type into the converter's memory block, and assign
    // its handle to the converter's convertible variable.  The C++
    // container is populated by passing the begin and end iterators of
    // the python object to the container's constructor.
    data->convertible = new (storage) Container(
      iterator(python::object(handle)), // begin
      iterator());                      // end
  }
};

#if defined(HAVE_MPI)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(dt_compare, ambit::tensor::template CyclopsTensor<double>::compare, 1, 2);
#endif // defined(MPI)

BOOST_PYTHON_MODULE(ambit)
{
    class_<std::vector<int> >("IntVec")
        .def(vector_indexing_suite<std::vector<int>, true >())
    ;

#if defined(HAVE_MPI)
    iterable_converter()
        // Built-n type.
        .from_python<std::vector<int> >()
        .from_python<std::vector<tkv_pair<double> > >()
    ;

    class_<tkv_pair<double> >("TkvPair", "Tensor key/value pair")
        .def(init<>())
        .def(init<tkv_pair<double> >())
        .def(init<key, double>())
        .def_readwrite("k", &tkv_pair<double>::k)
        .def_readwrite("d", &tkv_pair<double>::d)
    ;

    class_<std::vector<tkv_pair<double> > >("TkvVec")
        .def(vector_indexing_suite<std::vector<tkv_pair<double> > >())
    ;

    class_<ambit::util::World>("World", "World communicator")
        .def_readonly("rank", &ambit::util::World::rank)
        .def_readonly("nproc", &ambit::util::World::nproc)
    ;

    typedef void   (ambit::tensor::CyclopsTensor<double>::*dt_mult)(double, const ambit::tensor::CyclopsTensor<double>&, const std::string&, const ambit::tensor::CyclopsTensor<double>&, const std::string&, double, const std::string&);
    typedef void   (ambit::tensor::CyclopsTensor<double>::*dt_sum)(double, const ambit::tensor::CyclopsTensor<double>&, const std::string&, double, const std::string&);
    typedef void   (ambit::tensor::CyclopsTensor<double>::*dt_scale)(double, const std::string&);
    typedef double (ambit::tensor::CyclopsTensor<double>::*dt_dot)(const ambit::tensor::CyclopsTensor<double>&, const std::string&, const std::string&) const;
    typedef void   (ambit::tensor::CyclopsTensor<double>::*dt_invert)(double, const ambit::tensor::CyclopsTensor<double>&, double);
    typedef void   (ambit::tensor::CyclopsTensor<double>::*dt_div)(double, const ambit::tensor::CyclopsTensor<double>&, const ambit::tensor::CyclopsTensor<double>&, double);
    typedef std::vector<tkv_pair<double> > (ambit::tensor::CyclopsTensor<double>::*dt_get_local_data)() const;
    typedef void   (ambit::tensor::CyclopsTensor<double>::*dt_write0)();
    typedef void   (ambit::tensor::CyclopsTensor<double>::*dt_write1)(const std::vector<tkv_pair<double> >&);
//    typedef void   (ambit::tensor::CyclopsTensor<double>::*dt_write2)(double, double, const std::vector<tkv_pair<double> >&);

    class_<ambit::tensor::CyclopsTensor<double> >("Tensor", "Distributed tensor", no_init)
        .def(init<const std::string&, ambit::util::World&, const std::vector<int>&, const std::vector<int>&>())
        .def(init<const std::string&, ambit::util::World&, const std::vector<int>&, const std::vector<int>&, bool>())
        .def(init<const ambit::tensor::CyclopsTensor<double>&>())
        .def("print_out", &ambit::tensor::CyclopsTensor<double>::print, "Print out the tensor")
        .def("fill_with_random_data", &ambit::tensor::CyclopsTensor<double>::fill_with_random_data, "Fills tensor with random data")
        .def("get_lengths", &ambit::tensor::CyclopsTensor<double>::get_lengths, return_value_policy<copy_const_reference>(), "Returns the lengths of the tensor")
        .def("get_symmetry", &ambit::tensor::CyclopsTensor<double>::get_symmetry, return_value_policy<copy_const_reference>(), "Returns the symmetries of the tensors")
        .def("contract", dt_mult(&ambit::tensor::CyclopsTensor<double>::mult), "Performs tensor contraction")
        .def("sort", &ambit::tensor::CyclopsTensor<double>::sort, "Sort tensor")
        .def("sum", dt_sum(&ambit::tensor::CyclopsTensor<double>::sum), "Performs tensor summation")
        .def("scale", dt_scale(&ambit::tensor::CyclopsTensor<double>::scale), "Performs tensor scaling")
        .def("dot", dt_dot(&ambit::tensor::CyclopsTensor<double>::dot), "Dots a tensor with this")
        .def("compare", &ambit::tensor::CyclopsTensor<double>::compare, dt_compare("Compares two tensors"))
        .def("invert", dt_invert(&ambit::tensor::CyclopsTensor<double>::invert), "Inverts the tensor")
        .def("div", dt_div(&ambit::tensor::CyclopsTensor<double>::div), "Divides one tensor by another")
        .def("resize", &ambit::tensor::CyclopsTensor<double>::resize, "Resize the tensor")
        .def("read_local", dt_get_local_data(&ambit::tensor::CyclopsTensor<double>::read_local), "Retrieve node tensor data")
        .def("write", dt_write0(&ambit::tensor::CyclopsTensor<double>::write), "Writes tensor data, remotely, if needed")
        .def("write", dt_write1(&ambit::tensor::CyclopsTensor<double>::write), "Writes tensor data, remotely, if needed")
//        .def("write", dt_write2(&ambit::tensor::CyclopsTensor<double>::write), "Writes tensor data, remotely, if needed")
    ;
#endif // defined(MPI)

}

int main(int argc, char** argv)
{
#if defined(HAVE_MPI)
    MPI::Init(argc, argv);
#endif // defined(MPI)

#if PY_MAJOR_VERSION == 2
    if (PyImport_AppendInittab(strdup("ambit"), initambit) == -1)
#else
    if (PyImport_AppendInittab(strdup("ambit"), PyInit_ambit) == -1)
#endif
    {
        fprintf(stderr, "Unable to register ambit with Python.\n");
        abort();
    }

    Py_InitializeEx(0);
#if PY_MAJOR_VERSION == 2
    Py_SetProgramName(strdup("ambit"));
#else
    Py_SetProgramName(L"ambit");
#endif

    std::string dirname = ROOT_SRC_DIR "/lib";
    // python library path
    PyObject *path, *sysmod, *str;
    PY_TRY(sysmod , PyImport_ImportModule("sys"));
    PY_TRY(path   , PyObject_GetAttrString(sysmod, "path"));
 #if PY_MAJOR_VERSION == 2
    PY_TRY(str    , PyString_FromString(dirname.c_str()));
 #else
    PY_TRY(str    , PyUnicode_FromString(dirname.c_str()));
 #endif

    // Append to the path list
    PyList_Append(path, str);

    Py_DECREF(str);
    Py_DECREF(path);
    Py_DECREF(sysmod);


    std::string data;
    if (argc == 2) {
        std::ifstream infile(argv[1]);
        std::string line;
        std::stringstream stream;
        while (std::getline(infile, line)) {
            stream << line << std::endl;
        }
        data = stream.str();
    }
    else
        data = interactive;

    PyRun_SimpleString(data.c_str());

#if defined(HAVE_MPI)
    MPI::Finalize();
#endif // defined MPI

    return 0;
}

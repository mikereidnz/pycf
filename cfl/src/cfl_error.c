/*
    Copyright (C) 2014- Sebastian Horvath (sebastian.horvath@gmail.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/* Single definition of the global error handler and helpers.
 *
 * Previously these lived in cfl_error.h with `static` linkage, which
 * gave each translation unit its own copy of the handler pointer.  As
 * a result, cfl_set_error_handler() only affected the TU from which it
 * was called, while CFL_ERROR_* macros expanded in other TUs continued
 * to use the default printf handler.  See finding F-008 in
 * plan/audit_2026-04-27_171732_report.md.
 */

#include "cfl_error.h"

cfl_error_handler_t _cfl_error_handler = NULL;

void _cfl_default_error_handler(const char *func, const char *file,
    int line, const char *message)
{
  printf("CFL_ERROR in function %s of %s, line %i: %s\n", func, file, line,
      message);
}

void cfl_set_error_handler(cfl_error_handler_t handler)
{
  _cfl_error_handler = handler;
}

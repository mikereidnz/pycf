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

/* Error handling with optional logging callback system.
 *
 * Supports two modes:
 * 1. Default: Errors print to stdout via printf()
 * 2. Custom: Registered callback function handles all error messages
 *
 * Use cfl_set_error_handler() to register custom error callback.
 * Call cfl_set_error_handler(NULL) to restore default printf() behavior.
 *
 * This allows production deployments to redirect errors to:
 * - Log files
 * - Syslog
 * - Custom error tracking systems
 * - Application-specific error handlers
 */

#ifndef _CFL_ERROR_H_
#define _CFL_ERROR_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Error handler callback signature.
 *
 * Parameters:
 *   func: Function name where error occurred
 *   file: Source file name
 *   line: Line number
 *   message: Error message
 *
 * The callback is responsible for all output (printing, logging, etc).
 */
typedef void (*cfl_error_handler_t)(const char *func, const char *file,
    int line, const char *message);

/* Global error handler pointer.
 *
 * Defined in cfl_error.c so that all translation units share a single
 * handler.  A previous version declared this `static` in the header,
 * which gave each TU its own copy and meant cfl_set_error_handler()
 * only updated the caller's TU.  See plan/audit_2026-04-27_171732_report.md
 * (finding F-008).
 */
extern cfl_error_handler_t _cfl_error_handler;

/* Default error handler: print to stdout.  Defined in cfl_error.c. */
void _cfl_default_error_handler(const char *func, const char *file,
    int line, const char *message);

/* Register custom error handler.  Defined in cfl_error.c.
 *
 * Parameters:
 *   handler: Function pointer or NULL
 *
 * If handler is NULL, restores default printf() behavior.
 * If handler is non-NULL, all error messages use the handler.
 *
 * Thread-safe for single assignment; assumes no concurrent handler registration.
 */
void cfl_set_error_handler(cfl_error_handler_t handler);

/* Internal macro to call error handler (default or custom).
 *
 * Routes to registered handler if available, else uses default printf().
 */
#define CFL_INVOKE_ERROR_HANDLER(message) do { \
  if (_cfl_error_handler != NULL) { \
    _cfl_error_handler(__func__, __FILE__, __LINE__, message); \
  } else { \
    _cfl_default_error_handler(__func__, __FILE__, __LINE__, message); \
  } \
} while(0)

/* Error macros using the logging system. */

#define CFL_ERROR_VAL(message, value) { \
  CFL_INVOKE_ERROR_HANDLER(message); \
  return value; \
}

#define CFL_ERROR_NULL(message) { \
  CFL_INVOKE_ERROR_HANDLER(message); \
  return NULL; \
}

#define CFL_ERROR_VOID(message) { \
  CFL_INVOKE_ERROR_HANDLER(message); \
  return ; \
}

#endif /* _CFL_ERROR_H_ */

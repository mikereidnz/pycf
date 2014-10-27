/*
    Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
 
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

/*
 * @file    cfl_error.h
 * @brief   Macros for handling error messages. 
 */

#ifndef _CFL_ERROR_H_
#define _CFL_ERROR_H_

#define CFL_ERROR_VAL(message, value) { \
  printf("CFL_ERROR in function %s of %s, line %i: %s\n",__func__, __FILE__,\
      __LINE__, message); \
  return value; \
}

#define CFL_ERROR_NULL(message) { \
  printf("CFL_ERROR in function %s of %s, line %i: %s\n",__func__, __FILE__,\
      __LINE__, message); \
  return NULL; \
}

#define CFL_ERROR_VOID(message) { \
  printf("CFL_ERROR in function %s of %s, line %i: %s\n",__func__, __FILE__,\
      __LINE__, message); \
  return ; \
}

#endif /* _CFL_ERROR_H_ */

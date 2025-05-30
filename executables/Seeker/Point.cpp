/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 * Copyright (c) 2025 Chapoly1305, William Flores
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Point.h"
#include <iomanip>
#include <sstream>

Point::Point() {}

Point::Point(const Point &p) {
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx, Int *cy, Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
  x.Set(cx);
  z.Set(cz);
}

void Point::Clear() {
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Int *cx, Int *cy, Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {}

void Point::Set(Point &p) {
  x.Set(&p.x);
  y.Set(&p.y);
}

bool Point::isZero() { return x.IsZero() && y.IsZero(); }

void Point::Reduce() {

  Int i(&z);
  i.ModInv();
  x.ModMul(&x, &i);
  y.ModMul(&y, &i);
  z.SetInt32(1);
}

bool Point::equals(Point &p) { return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z); }

std::string Point::toString() {
    std::ostringstream oss;
    oss << "04:";
    oss << std::setfill('0') << std::setw(56) << x.GetBase16();
    oss << ":";
    oss << std::setfill('0') << std::setw(56) << y.GetBase16();
    return oss.str();
}
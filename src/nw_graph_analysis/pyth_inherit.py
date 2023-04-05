class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def peri(self):
        return 2*(self.length + self.width)


class Sq:
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side

    def peri(self):
        return 4 * self.side


class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)


class Cube(Square):
    def surface_area(self):
        fac_area = super().area()
        return 6 * fac_area

    def vol(self):
        fac_area = super().area()
        return fac_area * self.length


class Sqr(Rectangle):
    def __init__(self, length):
        super(Sqr, self).__init__(length, length)


class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height

class RightPyramid(Square, Triangle):
    def __init__(self, base, slh, height):
        self.base = base
        self.slh = slh
        super(RightPyramid, self).__init__(self.base)

    def area(self):
        base_area = super().area()
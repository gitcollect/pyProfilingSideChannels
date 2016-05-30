import csv
#import pickle
#import logging
#import threading
import os
import random
from itertools import chain
from scipy import zeros, resize, ravel, asarray
import scipy

# file extensions
known_extensions = {
    '.csv': 'csv',
    '.mat': 'matlab',
    '.txt': 'ascii',
    '.pkl': 'pickle'}


class OutOfSyncError(Exception):
    pass


class VectorFormatError(Exception):
    pass


class NoLinkedFieldsError(Exception):
    pass


def abstractMethod():
    """Only invoked when it is not implemented"""
    raise NotImplementMethod('Method no implement ')


def formatFromExtension(filename):
    """Split the filename to get the extension"""
    root, ext = os.path.splitext(filename)
    if not ext:
        return None
    try:
        format = known_extensions[ext]
    except KeyError:
        format = None
        return format


class IOAdapter(object):
    """ Reader/Writer handler"""

    def _save_csv(self, fileobject, **kwargs):
        fieldnames = self._getFieldNames()
        writer = csv.DictWriter(fileobject, fieldnames)
        writer.writeheader()
        writer.writerows(self._getDictRows())

    @classmethod
    def _load_csv(cls, fileobject):
        """ """
        reader = csv.DictReader(fileobject)
        return cls._reconstruct(reader)

    def _saveFileLike(self, fileobject, format=None, **kwargs):
        """Save obj to the file, format can be pickle, csv or txt"""
        format = 'csv' if format is None else format
        save = getattr(self, "_save_%s" % format, None)
        if save is None:
            raise ValueError("Unknown format '%s' ." % format)
        save(fileobject, **kwargs)

    @classmethod
    def _loadFileLike(cls, fileobject, format=None):
        """Load object to a given file"""
        format = 'csv' if format is None else format
        load = getattr(cls, "_load_%s" % format, None)
        if load is None:
            raise ValueError("Unknown format '%s'." % format)
        return load(fileobject)

    def saveFile(self, filename, format=None, **kwargs):
        if not format:
            format = formatFromExtension(filename)
            with open(filename, 'wb') as fileobject:
                self._saveFileLike(fileobject, format, **kwargs)

    @classmethod
    def loadFile(cls, filename, format=None, **kwargs):
        if not format:
            format = formatFromExtension(filename)
            with open(filename, 'rb') as fileobject:
                clsobj = cls._loadFileLike(fileobject, format)
                clsobj.filename = filename
                return clsobj

    def _getFieldNames(self):
        return abstractMethod()

    def _getDictRows(self):
        return abstractMethod()

    @classmethod
    def _reconstruct(cls, reader):
        return abstractMethod()


class DataSet(IOAdapter):
    """DataSet is a general base class for other data set classes"""

    def __init__(self):
        self._data = {}
        self._endmark = {}
        self._link = []
        self._index = 0
        # row vectors returned by getLinked can have different formats:
        # '1d'       example: array([1, 2, 3])
        # '2d'       example: array([[1, 2, 3]])
        # 'list'     example: [1, 2, 3]

        # vectorformat is a property
        self.vectorformat = 'none'

    def __str__(self):
        """Return a string representation of a dataset"""
        s = ""
        for key in self._data:
            s = s + key + ": dim" + \
                str(self._data[key].shape) + "\n" + \
                str(self._data[key][:self._endmark[key]]) + "\n\n"

        return s

    def __getitem__(self, field):
        """Return the arrray of given field of the object"""
        return self.getField(field)

    def __iter__(self):
        self.reset()
        while not self.endOfData():
            yield self.getLinked()

    def __len__(self):
        """Return the length of object"""
        return self.getLength()

    def _getVectorFormat(self):
        return self._vectorformat

    def _setVectorFormat(self, vfmt):
        """Determine which format to use for returning vectors. Use the property vectorformat.
        :key type: possible types are '1d', '2d', 'list'
        '1d' - example: array([1,2,3])
        '2d' - example: array([[1,2,3]])
        'list' - example: [1,2,3]
        'none' - no conversion
        """
        switch = {
            '1d': self._convertArray1d,
            '2d': self._convertArray2d,
            'list': self._convertList,
            'none': lambda x: x
        }
        try:
            self._convert = switch[vfmt]
            self._vectorformat = vfmt
        except KeyError:
            raise VectorFormatError(
                "vector format must be 1d, 2d, list. given %s" % vfmt)

    vectorformat = property(_getVectorFormat, _setVectorFormat,
                            None, "vectorformat can be 1d or 2d and list")

    def _convertList(self, vector):
        """Convert the incoming vector to a python list"""
        return ravel(vector).tolist()

    def _convertArray1d(self, vector):
        return ravel(vector)

    def _convertArray2d(self, vector, column=False):
        """Converts the incoming `vector` to a 2d vector with shape (1,x),or (x,1) if `column` is set,
        where x is the number of elements."""
        a = asarray(vector)
        sh = a.shape
        if len(sh) == 0:
            # while 'a' is an empty vector
            sh = (1,)
        if len(sh) == 1:
            # use reshape to add extra dimension
            if column:
                return a.reshape((sh[0], 1))
            else:
                return a.reshape((1, sh[0]))
        else:
            # vector is not 1d, return a without change
            return a

    def addField(self, field, dim):
        """Add a field to DataSet"""
        self._data[field] = zeros((0, dim), float)
        self._endmark[field] = 0

    def setField(self, field, arr):
        arr_array = asarray(arr)
        self._data[field] = arr_array
        self._endmark[field] = arr_array.shape[0]

    def getField(self, field):
        """Return the entire field given by `field` as an array or list,
        depending on user settings."""
        # Note: field_data should always be a np.array, so this will never
        # actually clone a list (performances are O(1)).
        field_data = self._data[field][:self._endmark[field]]
        # Convert to list if requested.
        if self.vectorformat == 'list':
            return field_data.tolist()
        return field_data

    def hasField(self, field):
        return field in self._data

    def getFieldNames(self):
        return list(self._data.keys())

    def _getFieldNames(self):
        return self._link

    def convertField(self, field, newtype):
        """Convert the given field to a different data type."""
        try:
            self.setField(field, self._data[field].astype(newtype))
        except KeyError:
            raise KeyError('convert field %s not found' % field)

    def linkFields(self, linklist):
        """Link the fields in 'linklist' and the length of them must be identical. """
        length = self._data[linklist[0]].shape[0]
        for field in linklist:
            if self._data[field].shape[0] != length:
                raise OutOfSyncError
        self._link = linklist

    def unlinkFields(self, unlinklist=None):
        """Remove the fields in 'unlinklist', unless the fields are not linked acturally.
        If 'unlinklist' is set to 'None', all the fields in linklist will be removed. """
        #link = self._link
        if unlinklist is not None:
            for field in unlinklist:
                if field in self._link:
                    self._link.remove()
                    #self._link = link
                else:
                    self._link = []

    def getDimension(self, field):
        """Return the dimension(number of cols) of field. """
        try:
            field_data = self._data[field]
        except KeyError:
            raise KeyError("dataset field %s not found" % field)
        # if 'field_data' is an empty array or 1d array just return 0 or 1
        return len(field_data.shape) if len(field_data.shape) < 2 else field_data.shape[1]

    def getLength(self):
        """Return the length of the linked fields. If none of them are linked, return the maximum one. """
        return self._endmark[max(self._endmark)] if self._link == [] else self._endmark[self._link[0]]

    def _resizeArray(self, arr):
        """Resize the array 'arr', double size each time. """
        shape = list(arr.shape)
        shape[0] = (shape[0] + 1) * 2
        return resize(arr, shape)

    def _resize(self, field):
        """Resize the buf of 'field', depends on the value of 'field'. """
        if field is not None:
            fields = [field]
        elif self._link != []:
            fields = self._link
        else:
            fields = self._data
            for f in fields:
                self._data[f] = self._resizeArray(self._data[f])

    def _appendField(self, field, row):
        """Add a `row` to the dataset under `field`. """
        if self._data[field].shape[0] <= self._endmark[field]:
            self._resize(field)
            self._data[field][self._endmark[field], :] = row
            self._endmark[field] += 1

    def appendUnlinkedField(self, field, row):
        """Add a `row` to the unlinked `field`. """
        if field in self._link:
            raise OutOfSyncError
        self._appendField(field, row)

    def appendLinkedFields(self, *args):
        """Add a `row` to all the linked fields. s.t. len(args) == len(linklist). """
        assert len(args) == len(self._link)
        for index, field in enumerate(self._link):
            self._appendField(field, args[index])

    def getLinked(self, index=None):
        """Access the linked fields.
        If called with `index`, the appropriate line consisting of all linked
        fields is returned and the internal marker is set to the next line.
        Otherwise the marked line is returned and the marker is moved to the
        next line. """
        if self._link == []:
            raise NoLinkedFieldsError(
                'The DataSet doesnt have any linked fields. ')
        if index == None:
            index = self._index
            self._index += 1
        else:
            self._index = index + 1
            if index >= self.getLength():
                raise IndexError('index out of DataSet size. ')
        return self._convert([self._data[field][index] for field in self._link])

    def _getDictRows(self):
        """Return the rows of DataSet of linkFields, in form of dict. """
        if self._link == []:
            raise NoLinkedFieldsError(
                'No linked fields existing, cannot use csv.writer. '
            )
        return [{field: self._data[field][index] for field in self._link} for index in range(len(self))]

    def endOfData(self):
        return self._index == self.getLength()

    def reset(self):
        """Reset the marker to the first line."""
        self._index = 0

    def clear(self, unlinked=False):
        """Clear the DataSet, only the `linked'`field will be deleted, unless the unlinked equals true.
        If there are no `linkded` field exist, the whole DataSet will be deleted. """
        self.reset()
        todelete = self._link
        if todelete == [] or unlinked:
            # iterate over all field
            todelete = self._data
            for field in todelete:
                shape = list(self._data[field].shape)
                shape[0] = 0
                self._data[field] = zeros(shape)
                self._endmark[field] = 0

    @classmethod
    def _reconstruct(cls, reader):
        """Construct an instance object from `csv.reader`. """
        obj = cls()
        buf = {}
        fieldnames = reader.fieldnames
        for field in fieldnames:
            buf[field] = []

        for row in reader:
            for field in fieldnames:
                buf[field] += [row[field]]

        obj._data = {field: asarray(buf[field], float) for field in fieldnames}
        obj.vectorformat = 'list'
        obj._link = reader.fieldnames
        obj._endmark = {field: len(obj._data[field]) for field in fieldnames}

        return obj

    def _save_csv(self, fileobject, compact=False):
        """Save DataSet as csv, removing empty space if desired"""
        if compact:
            temp = self._data[field][0:self._endmark[field] + 1, :]
            self.setField(field, temp)
            # FIXME : Deprecated: IOAdapter.save_pickle(self, filename,
            # protocal)
            super(IOAdapter, self)._save_csv(fileobject)

    def __reduce__(self):
        def creater():
            obj = self.__class__()
            obj.vectorformat = self.vectorformat
            return obj
        args = tuple()
        state = {
            'data': self._data,
            'link': self._link,
            'endmark': self._endmark,
        }
        return creater, args, state, iter([]), iter({})

    def copy(self):
        """Return a deep copy. """
        import copy
        return copy.deepcopy(self)

    def batches(self, field, n, permutation=None):
        """Yield batches of the size of n of DataSet.

        A single batch is an array of with dim columns and n rows. The last
        batch is possibly smaller.

        If permutation is given, batches are yielded in the corresponding
        order.
        """
        # First calculate how many batches we will have
        full_batches, rest = divmod(len(self), n)
        number_of_batches = full_batches if rest == 0 else full_batches + 1

        # Get the iterators
        start_iterator = (i * n for i in range(number_of_batches))
        end_iterator = ((i + 1) * n for i in range(number_of_batches))
        # The last end index is the last element of the list (last batch
        # might not be filled completely)
        end_iterator = chain(end_iterator, [len(self)])
        # Now generate the real indexes by combining them:
        indexes = list(zip(start_iterator, end_iterator))
        if permutation is not None:
            # Shuffle them
            indexes = [indexes[i] for i in permutation]
            for st, ed in indexes:
                yield self._data[field][st:ed]

    def randomBatches(self, field, n):
        permutation = random.shuffle(list(len(self)))
        return self.batches(field, n, permutation)

    def replaceNaNsByMeans(self):
        """Replace all not-a-number entries in the dataset by the means of the
        corresponding column."""
        for field in self._data.keys():
            means = scipy.nansum(
                self._data[field][:len(self)], axis=0) / len(self)
            for i in range(len(self)):
                for j in range(self.getDimension(field)):
                    if not scipy.isfinite(self._data[field][i, j]):
                        self._data[field][i, j] = means[j]


##
x = DataSet()
x.vectorformat = '1d'
x.addField('lab1', 1)
x.addField('lab2', 1)
x.setField('lab1', [0.112, 0.0000, 0.333, 0.998])

#print x

x.replaceNaNsByMeans

y = DataSet.loadFile('/Users/pushbeni/Dropbox/r/final/Features_Test.csv')
y.vectorformat = '2d'
print y._index
print y.getLinked().__class__
#print y['V10']
#print y

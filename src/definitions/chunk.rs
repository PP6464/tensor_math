use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};

/// A chunk of data from a tensor-like or matrix-like value. For densely packed
/// tensors and matrices the data is stored as a borrowed slice; for slices the
/// data is materialised into a `Vec` because the elements are not guaranteed
/// to be contiguous in memory.
pub enum Chunk<'a, T> {
    Contiguous(&'a [T]),
    NonContiguous(Vec<&'a T>),
}

impl<'a, T> Chunk<'a, T> {
    /// The length of the chunk
    pub fn len(&self) -> usize {
        match self {
            Chunk::Contiguous(s) => s.len(),
            Chunk::NonContiguous(v) => v.len(),
        }
    }

    /// An iterator over references to the elements in the chunk
    pub fn iter(&'a self) -> ChunkIter<'a, T> {
        match self {
            Chunk::Contiguous(s) => ChunkIter::Contiguous(s.iter()),
            Chunk::NonContiguous(v) => ChunkIter::NonContiguous(v.iter().copied()),
        }
    }

    /// A parallel iterator over references to the elements in the chunk
    pub fn par_iter(&'a self) -> ParChunkIter<'a, T>
    where
        T: Send + Sync,
    {
        match self {
            Chunk::Contiguous(s) => ParChunkIter::Contiguous(s.par_iter()),
            Chunk::NonContiguous(v) => ParChunkIter::NonContiguous(v.par_iter().copied()),
        }
    }
}

pub enum ChunkIter<'a, T> {
    Contiguous(std::slice::Iter<'a, T>),
    NonContiguous(std::iter::Copied<std::slice::Iter<'a, &'a T>>),
}

impl<'a, T> Iterator for ChunkIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ChunkIter::Contiguous(it) => it.next(),
            ChunkIter::NonContiguous(it) => it.next(),
        }
    }
}

impl<'a, T> DoubleEndedIterator for ChunkIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            ChunkIter::Contiguous(it) => it.next_back(),
            ChunkIter::NonContiguous(it) => it.next_back(),
        }
    }
}

pub enum ParChunkIter<'a, T> {
    Contiguous(rayon::slice::Iter<'a, T>),
    NonContiguous(rayon::iter::Copied<rayon::slice::Iter<'a, &'a T>>),
}

impl<'a, T: Sync + 'a> ParallelIterator for ParChunkIter<'a, T> {
    type Item = &'a T;

    fn drive_unindexed<C: UnindexedConsumer<Self::Item>>(self, consumer: C) -> C::Result {
        match self {
            ParChunkIter::Contiguous(it) => it.drive_unindexed(consumer),
            ParChunkIter::NonContiguous(it) => it.drive_unindexed(consumer),
        }
    }

    fn opt_len(&self) -> Option<usize> {
        Some(<Self as IndexedParallelIterator>::len(self))
    }
}

impl<'a, T: Sync + 'a> IndexedParallelIterator for ParChunkIter<'a, T> {
    fn len(&self) -> usize {
        match self {
            ParChunkIter::Contiguous(it) => it.len(),
            ParChunkIter::NonContiguous(it) => it.len(),
        }
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        match self {
            ParChunkIter::Contiguous(it) => it.drive(consumer),
            ParChunkIter::NonContiguous(it) => it.drive(consumer),
        }
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        match self {
            ParChunkIter::Contiguous(it) => it.with_producer(callback),
            ParChunkIter::NonContiguous(it) => it.with_producer(callback),
        }
    }
}

/// A mutable chunk of data from a tensor-like or matrix-like value. For densely packed
/// tensors and matrices the data is stored as a borrowed slice; for slices the
/// data is materialised into a `Vec` because the elements are not guaranteed
/// to be contiguous in memory.
pub enum ChunkMut<'a, T> {
    Contiguous(&'a mut [T]),
    NonContiguous(Vec<&'a mut T>),
}

impl <'a, T> ChunkMut<'a, T> {
    /// The length of the chunk
    pub fn len(&self) -> usize {
        match self {
            ChunkMut::Contiguous(s) => s.len(),
            ChunkMut::NonContiguous(v) => v.len(),
        }
    }

    /// An iterator over mutable references to the elements in the chunk
    pub fn iter_mut(&'a mut self) -> ChunkMutIter<'a, T> {
        match self {
            ChunkMut::Contiguous(s) => ChunkMutIter::Contiguous(s.iter_mut()),
            ChunkMut::NonContiguous(v) => ChunkMutIter::NonContiguous(v.iter_mut().map(deref_mut_ref)),
        }
    }

    /// A parallel iterator over mutable references to the elements in the chunk
    pub fn par_iter_mut(&'a mut self) -> ParChunkMutIter<'a, T>
    where
        T: Send + Sync,
    {
        match self {
            ChunkMut::Contiguous(s) => ParChunkMutIter::Contiguous(s.par_iter_mut()),
            ChunkMut::NonContiguous(v) => ParChunkMutIter::NonContiguous(v.par_iter_mut().map(deref_mut_ref)),
        }
    }
}

fn deref_mut_ref<'b, 'a: 'b, T>(r: &'b mut &'a mut T) -> &'b mut T {
    &mut **r
}

pub(crate) enum ChunkMutIter<'b, T> {
    Contiguous(std::slice::IterMut<'b, T>),
    NonContiguous(
        std::iter::Map<std::slice::IterMut<'b, &'b mut T>, fn(&'b mut &'b mut T) -> &'b mut T>,
    ),
}

impl<'b, T> Iterator for ChunkMutIter<'b, T> {
    type Item = &'b mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ChunkMutIter::Contiguous(it) => it.next(),
            ChunkMutIter::NonContiguous(it) => it.next(),
        }
    }
}

pub(crate) enum ParChunkMutIter<'b, T: Send> {
    Contiguous(rayon::slice::IterMut<'b, T>),
    NonContiguous(
        rayon::iter::Map<rayon::slice::IterMut<'b, &'b mut T>, fn(&'b mut &'b mut T) -> &'b mut T>,
    ),
}

impl<'b, T: Send + 'b> ParallelIterator for ParChunkMutIter<'b, T> {
    type Item = &'b mut T;

    fn drive_unindexed<C: UnindexedConsumer<Self::Item>>(self, consumer: C) -> C::Result {
        match self {
            ParChunkMutIter::Contiguous(it) => it.drive_unindexed(consumer),
            ParChunkMutIter::NonContiguous(it) => it.drive_unindexed(consumer),
        }
    }

    fn opt_len(&self) -> Option<usize> {
        Some(<Self as IndexedParallelIterator>::len(self))
    }
}

impl<'b, T: Send + 'b> IndexedParallelIterator for ParChunkMutIter<'b, T> {
    fn len(&self) -> usize {
        match self {
            ParChunkMutIter::Contiguous(it) => it.len(),
            ParChunkMutIter::NonContiguous(it) => it.len(),
        }
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        match self {
            ParChunkMutIter::Contiguous(it) => it.drive(consumer),
            ParChunkMutIter::NonContiguous(it) => it.drive(consumer),
        }
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        match self {
            ParChunkMutIter::Contiguous(it) => it.with_producer(callback),
            ParChunkMutIter::NonContiguous(it) => it.with_producer(callback),
        }
    }
}

# coding=utf-8

import io
import re
from collections import defaultdict
from itertools import chain
from typing import List, Optional, Type, Union





class Feature:
    """
    A class of genomic features defined by a half-open interval. Locations are 0-based.
    """

    DUMMY_NAME = "."

    @staticmethod
    def from_string(line, transform=lambda x: x):
        columns = re.split("[\t ]+", line.rstrip())

        assert (
            len(columns) >= 3
        ), "Invalid BED file supplied: at least three columns are expected. Please, check carefully that the correct input type was selected."
        assert re.match(
            "[0-9]+", columns[1]
        ), "Invalid BED file supplied: second column must contain integers."
        assert re.match(
            "[0-9]+", columns[2]
        ), "Invalid BED file supplied: third column must contain integers."

        name = Feature.DUMMY_NAME if len(columns) == 3 else transform(columns[3])

        try:
            score = float(re.sub(",", ".", columns[4])) if len(columns) >= 5 else None
        except ValueError:
            raise AssertionError(
                "Invalid BED file supplied: fifth column must contain floating point numbers (score)."
            )

        strand = columns[5] if len(columns) >= 6 else None

        assert not strand or strand in (
            "+",
            "-",
            ".",
            "?",
        ), "Invalid BED file supplied: sixth column must contain strand (+/-/?)."

        return Feature(
            columns[0], int(columns[1]), int(columns[2]), name, score, strand
        )

    def __init__(
        self, chromosome, start, end, name=DUMMY_NAME, score=None, strand=None
    ):
        assert chromosome.strip() != ""
        assert end >= start
        assert name.strip() != ""

        self.chromosome = chromosome
        self.interval = (start, end)
        self.name = name
        self.score = score
        self.strand = strand

    @property
    def start(self):
        return self.interval[0]

    @property
    def end(self):
        return self.interval[1]

    def __repr__(self):
        return "Feature({})".format(str(self).replace("\t", ","))

    def __str__(self):
        r = "{0:s}\t{1:d}\t{2:d}\t{3:s}".format(
            self.chromosome, self.interval[0], self.interval[1], self.name
        )

        if self.score and self.strand:
            r += "\t{0:f}\t{1:s}".format(self.score, self.strand)
        elif self.score:
            r += "\t{0:f}".format(self.score)
        elif self.strand:
            r += "\t0.0\t{0:s}".format(self.strand)
        return r

    def __len__(self):
        """Length of feature in base pairs."""
        return self.interval[1] - self.interval[0]

    def has_overlap_with(self, other):
        return (
            self.chromosome == other.chromosome
            and self.interval[0] < other.interval[1]
            and self.interval[1] > other.interval[0]
        )

    def __contains__(self, other):
        return (
            self.chromosome == other.chromosome
            and other.interval[0] >= self.interval[0]
            and other.interval[1] <= self.interval[1]
        )

    def get_overlap_in_bp_with(self, other):
        if not self.has_overlap_with(other):
            return 0

        return min(self.interval[1], other.interval[1]) - max(
            self.interval[0], other.interval[0]
        )


class FeatureSeq(object):
    """

    A sequence of features.

    """

    NAME_ATTRIBUTE = "name"
    SCORE_ATTRIBUTE = "score"

    @staticmethod
    def from_bed_file(file: Union[str, Type[io.IOBase]], transform=lambda x: x):
        if isinstance(file, io.IOBase):

            def _feature_iterator():
                for line in file:
                    yield Feature.from_string(line, transform)

        else:

            def _feature_iterator():
                with open(file, "r") as f:
                    for line in f:
                        yield Feature.from_string(line, transform)

        return FeatureSeq(_feature_iterator())

    def __init__(self, features_iterator):
        from interlap import InterLap
        # Fast discovery of overlapping intervals.
        self.chromosome2tree = defaultdict(InterLap)
        self.name2features = defaultdict(list)

        for feature in features_iterator:
            self.chromosome2tree[feature.chromosome].add(
                (
                    *feature.interval,
                    {
                        FeatureSeq.NAME_ATTRIBUTE: feature.name,
                        FeatureSeq.SCORE_ATTRIBUTE: feature.score,
                    },
                )
            )
            self.name2features[feature.name].append(feature)

    def __len__(self):
        return sum(map(len, self.name2features.values()))

    def __iter__(self):
        return chain.from_iterable(self.name2features.values())

    def __str__(self):
        return "\n".join(map(str, self))

    def get(self, name: str) -> "FeatureSeq":
        # Return a new list as a defensive copy mechanism.
        return FeatureSeq(self.name2features[name])

    def find(self, feature: Feature, fraction: Optional[float] = None) -> List[Feature]:
        from interlap import InterLap
        def filter4Fraction(overlap_feature):
            if not fraction:
                return True

            overlap_in_bp = float(overlap_feature.get_overlap_in_bp_with(feature))

            if len(feature) == 0:
                overlap_fraction_relative_to_feature = 0.0
            else:
                overlap_fraction_relative_to_feature = overlap_in_bp / len(feature)

            if len(overlap_feature) == 0:
                overlap_fraction_relative_to_overlap_feature = 0.0
            else:
                overlap_fraction_relative_to_overlap_feature = overlap_in_bp / len(
                    overlap_feature
                )

            return (
                max(
                    overlap_fraction_relative_to_feature,
                    overlap_fraction_relative_to_overlap_feature,
                )
                >= fraction
            )

        def toFeature(interval):
            return Feature(
                feature.chromosome,
                interval[0],
                interval[1],
                interval[2][FeatureSeq.NAME_ATTRIBUTE],
                interval[2][FeatureSeq.SCORE_ATTRIBUTE],
            )

        return list(
            filter(
                filter4Fraction,
                map(
                    toFeature,
                    self.chromosome2tree.get(feature.chromosome, InterLap()).find(
                        feature.interval
                    ),
                ),
            )
        )

    def intersection(
        self, other: "FeatureSeq", fraction: Optional[float] = None
    ) -> "FeatureSeq":
        def _feature_iterator(self, other):
            for feature1 in other:
                for feature2 in self.find(feature1, fraction):
                    yield feature2

        return FeatureSeq(_feature_iterator(self, other))

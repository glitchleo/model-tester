#!/usr/bin/python
# -*- coding: UTF-8 -*-

from abc import ABC, abstractmethod

class TrainerBase(ABC):
    @property
    def timestamp(self):
        return ""

    @abstractmethod
    def run(self):
        pass

    @property
    def name(self):
        return ""

    @property
    def config(self) -> dict:
        return {}

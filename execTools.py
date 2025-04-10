#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:58:47 2023

@author: nchevaug
"""
import cProfile


class Profile:
    doprofile = True
    pr = cProfile.Profile()

    @classmethod
    def enable(cls):
        if Profile.doprofile:
            Profile.pr.enable()

    @classmethod
    def disable(cls):
        if Profile.doprofile:
            Profile.pr.disable()

    @classmethod
    def print_stats(cls):
        if Profile.doprofile:
            Profile.pr.print_stats(sort="cumulative")

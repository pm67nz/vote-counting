#!/usr/bin/env python3

# Copyright 2023 Peter Maxwell
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of 
# the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.

from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from math import log10
from hashlib import md5
from base64 import urlsafe_b64encode 
from time import time
import re
import datetime

import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

@dataclass
class Ballots:
    ballots: list(float, list(int))
    num_candidates: int
    
    @property
    def votes(self):
        return sum(w for (w,bs) in self.ballots)
    
    def len(self):
        return len(self.ballots)

    def meek_distribute_votes(self, keep_factor):
        votes = np.zeros_like(keep_factor)
        for (weight, ballot) in self.ballots:
            remaining_weight = float(weight)
            for candidate in ballot:
                if remaining_weight <= 0: break
                fractional_vote = keep_factor[candidate] * remaining_weight
                votes[candidate] += fractional_vote
                remaining_weight -= fractional_vote
        return votes

    def compacted(self):
        weights = defaultdict(int)
        for (weight, ballot) in self.ballots:
            weights[tuple(ballot)] += weight
        ballots = [(weight, list(pattern)) for (pattern, weight) in weights.items()]
        return Ballots(ballots, self.num_candidates)

    def subset(self, reverse_map, without):
        considered = np.ones([len(reverse_map)], bool)
        considered[without] = False
        growmap = np.where(considered)[0]
        reverse_map = reverse_map[list(growmap)]
        shrinkmap = {b:a for (a,b) in enumerate(growmap)}
        ballots = [(weight, [shrinkmap[b] for b in ballot if considered[b]]) for (weight, ballot) in self.ballots]
        return (Ballots(ballots, len(reverse_map)), reverse_map)


def generate_meek_se(ballots, max_seats=None, withdrawn=[], profile={}, eta=1e-6, compact=True):
    """List Ranking by Sequential Exclusion using Meek STV with minimal complications,
    so no artificial limits on precision, and no complicated tie-breaking options.
    
    This function handles candidates as integer ordinals, as in the .blt file format,
    So there are no candidate names in here.
    
    ballots - Collection of sequences of candidates.
    withdrawn - Collection of candidates to ignore.
    """

    num_candidates = ballots.num_candidates
    order_of_exclusion = []
    reverse_map = np.arange(0, num_candidates, dtype=int)
    excluded = withdrawn
    nvc = num_candidates - len(withdrawn)
    
    seats = max_seats = min(max_seats or nvc, nvc)
    positions = list(range(nvc, 0, -1))

    efforts = [(p if p <= seats+1 else max(seats**0.5,seats*2-p)/2)**2 for p in range(nvc, 0, -1)] 
    efforts[0] = max(efforts[0], max(efforts)/2)
    steps_total = sum(efforts)
    steps_done = 0

    hopeful = np.ones([num_candidates], bool)
    elected = np.zeros([num_candidates], bool)
    keep_factor = np.ones([num_candidates], float)

    for (position, effort) in zip(positions, efforts):
        t0 = time()
        seats = min(position - 1, max_seats)
        method = 'STV-SE' if position <= max_seats else 'STV'
        
        if method == 'STV-SE':
            # Reset the winners as we are reducing the number of seats
            hopeful |= elected
            keep_factor[hopeful] = 1.0
            elected[:] = False

        if excluded:
            hopeful[excluded] = False
            keep_factor[excluded] = 0
            num_candidates -= len(excluded)
            if compact:
                # This makes no difference to the result, just an optimisation
                # (with a fairly unimpressive performance benefit).
                (ballots, reverse_map) = ballots.subset(reverse_map, excluded)
                ballots = ballots.compacted()
                assert num_candidates == len(reverse_map)
                keep = hopeful | elected
                hopeful = hopeful[keep]
                elected = elected[keep]
                keep_factor = keep_factor[keep]
            excluded = []
        
        message = f"{method} Electing {seats} out of {num_candidates}"
        steps_done += effort

        while any(hopeful):
            # Adjust the keep factors of elected candidates
            while True:
                votes = ballots.meek_distribute_votes(keep_factor)
                quota = sum(votes)/(seats+1)
                if all(np.abs(votes[elected]/quota - 1) < eta):
                    break
                keep_factor[elected] = np.minimum(1.0, keep_factor[elected] * quota / votes[elected])
            # At least one candidate must now be elected or excluded
            vacancies = seats - elected.sum()
            reached_quota = hopeful.nonzero()[0][votes[hopeful] > quota+eta]
            while reached_quota.size > vacancies:
                reached_quota = np.delete(reached_quota, [])
            if reached_quota.size > 0:
                elected[reached_quota] = True
                hopeful[reached_quota] = False
            else:
                lowest = hopeful.nonzero()[0][np.argmin(votes[hopeful])]
                excluded.append(lowest)
                keep_factor[lowest] = 0
                hopeful[lowest] = False
                yield (steps_done / steps_total, message, reverse_map[lowest])
                break
        t1 = time()
        profile['position'].append(position)
        profile['elapsed'].append(t1 - t0)

def read_ballots(f):
    """Read a .blt format file of ballots.
    File format is 
    <#candidates> <#winners>
    <weight> <1st choice of 1st voter> <2nd choice of 1st voter> ... 0
    <weight> <1st choice of 2nd voter> <2nd choice of 2nd voter> ... 0
    0
    1st candidate name
    2nd candidate name
    Title of the election
    
    All tokens in the first half of the file are integers.
    """
    undervoted = overvoted = weighted = 0
    candidates = seats = None
    withdrawn = []
    ballots = []
    while True:
        line = f.readline()
        assert line, 'Unexpected end of file'
        line = line.split('#')[0].strip()
        if '=' in line:
            line = line.replace('=', ' ')
            overvoted += 1
        words = line.split()
        if '-' in words:
            words = [w for w in words if w != '-']
            undervoted += 1
        row = [int(w) for w in words]
        if not row:
            pass
        elif row == [0]:
            break
        elif candidates is None:
            assert len(row) == 2, row
            (candidates, seats) = row
        elif max(row) < 0 and not (ballots or withdrawn):
            assert min(row) < 0, 'Line with only some negative numbers'
            withdrawn.extend(-c-1 for c in row)
        else:
            weight = row.pop(0)
            sentinal = row.pop()
            assert sentinal == 0, f'{sentinal} at end of line'
            assert min(row) > 0, 'Unexpected negative numbers'
            if weight != 1:
                weighted += 1
            ballots.append((weight, [c-1 for c in row]))
        
    names = [f.readline().strip().replace('"', '') for c in range(candidates)]
    assert '' not in names, 'Nameless candidate(s)'
    assert len(set(names)) == len(names), 'Duplicate candidate names'
    title = f.readline().strip().replace('"', '')
    
    parse_warnings = [f'{flag} ballot lines {msg}' for (flag, msg) in [
            (undervoted, 'included undervotes (skipped ranks). They will be renumbered.'),
            (overvoted, 'included overvotes (tied rankings). They will be tie-broken by order of appearance.'),
            (weighted, 'were weighted, ie: count as multiple ballots.')]
        if flag]

    return (ballots, names, title, parse_warnings, withdrawn, seats)


@st.cache_data(max_entries=1, show_spinner=False, scope="session")
def streamlit_meek_se(ballots, max_seats=None, withdrawn=[], eta=1e-6, compact=True):
    result = []
    my_bar = st.progress(0.0)
    profile_dict = defaultdict(list)
    for (progress, msg, candidate) in generate_meek_se(ballots, 
            max_seats=max_seats, withdrawn=withdrawn,
            profile=profile_dict, eta=eta, compact=compact):
        my_bar.progress(progress, text=msg)
        result.append(candidate)
    my_bar.empty()
    result.reverse()    
    return (result, pd.DataFrame(profile_dict))

def parse_result(text):
    lines = text.split('\n')
    result = []
    committed = False
    for line in lines:
        line = line.strip()
        if ':' in line or len(line)==0:
            continue
        words = re.split(' +|[\t,]', line)
        if re.match('[0-9]+', words[0]):
            del words[0]
        if words:
            result.append(' '.join(words))
    return result
            
def diff_chart(reference, current):
    list_names = ['Reference', 'Current']
    lists = [reference, current]
    
    dfs = [pd.DataFrame(
        {'cand': cands, 'list': [list_name]*len(cands), 'rank': range(1, len(cands)+1)})
        for (list_name, cands) in zip(list_names, lists)]

    base = alt.Chart(pd.concat(dfs))    
    x = alt.X('list:N', sort=list_names).axis(labelAngle=0, orient="top", title='')
    y = alt.Y('rank:O').axis().title('')

    chart = base.mark_line().encode(x=x, y=y, detail='cand')
    for (align, dx, list_name, cands) in zip(['right', 'left'], [-8, 8], list_names, lists):
        label = base.mark_text(align=align, dx=dx).encode(
                x=x, y=y, text=alt.Text('cand')).transform_filter(
                alt.datum.list == list_name)
        chart += label

    return chart

@dataclass
class Result:
    """Class for keeping track of results for diffing."""
    description: str
    ordered_names: list[str]
    
st.title('STV-SE ranking')
st.text('Generates an ordered Party List via Meek STV with Sequential Exclusion.', 
help = '''### STV with Sequential Exclusion

Given N candidates count the ballots as if for an ordinary N-1 seat STV election. Remove the lone non-winner from 
the set of candidates, placing them last in the overall result. Count the ballots as if for an N-2 seat STV election
and so on. See [Voting Matters 9](https://www.votingmatters.org.uk/ISSUE9/P4.HTM).

Equivalently, one STV election in which as soon all the seats are filled, the number of seats
is decremented (and so the quota raised and the winners reset) before continuing.

''')


uploaded_file = st.file_uploader("Choose a ballot file", 
        type=["txt", "blt"], 
        max_upload_size=10, 
        help="A [ballot file](https://opavote.com/help/overview#blt-file-format)")

if uploaded_file is None:
    candidates = []
    withdrawn = []
    file_seats = nc = 0
    ballots = None
else:
    data = uploaded_file.getvalue()
    checksum = urlsafe_b64encode(md5(data).digest()).decode('utf-8').strip('=')
    (ballots, candidates, title, parse_warnings, withdrawn, file_seats) = read_ballots(
            StringIO(data.decode("utf-8")))
    ballots = Ballots(ballots, len(candidates))
    if parse_warnings:
        st.warning('\n\n'.join(parse_warnings))
    nc = len(candidates)
    st.info(f'"{title}" - {nc} candidates and {ballots.votes} ballots.', icon=":material/summarize:")
    
withdrawn = st.multiselect("Select any candidates who have withdrawn", candidates, 
    default = [candidates[w] for w in withdrawn])
nvc = len(candidates) - len(withdrawn)

max_rank_opts, min_rank_opts = st.columns(2, gap="medium")

with max_rank_opts:
    st.text('Maximum number to rank', help="This many candidates will first be selected with ordinary "
            "STV, which will be faster than STV-SE and may give better results at the end of the list.")
    with st.container(horizontal=True):        
        def custom_seats_set():
            if st.session_state.seats is not None:
                st.session_state.seat_source = None
            elif st.session_state.seat_source is None:
                st.session_state.seat_source = 'All'
        
        qseats = st.query_params.get('seats')
        seats = None
        if qseats and 'seat_source' not in st.session_state:
            if qseats.lower() == 'file':
                st.session_state.seat_source = 'File'
            elif qseats.lower() == 'all':
                st.session_state.seat_source = 'All'
            else:
                st.session_state.seat_source = None
                seats = int(qseats)
                
        source_options = {'File': file_seats, 'All': nvc}
        source = st.radio('Number to rank', source_options, horizontal = True, 
            label_visibility = "collapsed",
            key = "seat_source",
            index = list(source_options).index(st.session_state.get('seat_source') or 'All'),
            captions = [str(v or '') for v in source_options.values()],
            help="This many candidates will first be selected with ordinary "
            "STV, which will be faster than STV-SE and may give better results at the end of the list.")
        if source is not None and ballots is not None:
            st.session_state.seats = source_options[source]
        seats = st.number_input("Seats", width=150, label_visibility="collapsed",
                key = "seats", value=seats, on_change=custom_seats_set, min_value = 1, max_value=100)
        seats = min(nvc, seats or nvc)
           
with min_rank_opts:
    st.text("Places to leave on top for leader(s)")
    reserved = st.radio("Places to add on top for leader(s) etc", [0,1,2], 
            index = int(st.query_params.get('top') or 0), label_visibility = "collapsed", horizontal = True)

ee = 6
compact = True

def calculate(profile=False):
    (result, profile_df) = streamlit_meek_se(ballots, seats, withdrawn=[candidates.index(name) for name in withdrawn], 
        eta=10**-ee, compact=compact)
    when = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M %Z')
    
    # PERFORMANCE PROFILING
    if profile:
        st.text('{:.2g} seconds'.format(sum(profile_df['elapsed'])))
        st.altair_chart(alt.Chart(profile_df).mark_bar().encode(
            x = alt.X('position:O', sort="descending"),
            y = alt.Y('elapsed:Q'),
        ), height=200)
        
    return (result, when)

result = None
if 'adv' in st.query_params:
    with st.container(border=True):
        with st.container(width="content", height="content", horizontal=True, vertical_alignment="center"):
            ee = st.slider('Digits of precision', key='ee', max_value=15, min_value=1, value=ee)
            st.space()
            compact = st.toggle('compact ballots', value=compact)        
        if ballots is not None:
            (result, when) = calculate(profile=True)
elif ballots is not None:
    (result, when) = calculate()
    
if result is not None:
    result = [None] * reserved + result
    seats += reserved
    
    # FORMATING RESULT   
    
    factoids = ''.join(f'{n}: {v}\n' for (n,v) in {
            'Ballot file md5 checksum': checksum,
            'Title': title, 
            'Ballots': ballots.votes,
            'Listed Candidates': nc,
            'Withdrawn': ', '.join(withdrawn) or 'none',
            'Remaining Candidates': nvc,
            'Counted at': when,
            'Precision': f'1e-{ee}',
        }.items())
            
    def generate_text(width, delim):
        lines = ['{Rank:{width}}{delim}{Candidate}'.format(width=width, delim=delim, 
                    Rank = i+1 if i < seats else '', 
                    Candidate = '' if c is None else candidates[c]) 
                for (i,c) in enumerate(result)]
        return '\n'.join(lines+[''])
    
    digits = int(log10(seats)) + 1
    formats = [
        ('Report', factoids, digits, '  ', 'txt'),
        ('CSV',  'Rank,Candidate', 1,  ',',  'csv'),
        ('TSV',  'Rank\tCandidate', 1, '\t', 'txt'),
        ]
    
    tabs = st.tabs([fmt[0] for fmt in formats] + ['Diff'])
    for (tab, fmt) in zip(tabs, formats):
        with tab:
            (name, head, width, delim, suffix) = fmt
            text = head + '\n' + generate_text(width, delim)
            with st.container(horizontal=True):
                st.code(text, language=None)
                st.download_button("", text, key=name, help=f"Download {name}", icon=":material/download:",
                        on_click="ignore", file_name=f'{title} Result.{suffix}')

    # DIFFING
    
    with tabs[-1]:
        with st.container(horizontal=True):
            reference = st.session_state.get('reference_result')
            current = Result(
                ordered_names = [candidates[c] for c in result if c is not None],
                description = factoids)
            with st.container(horizontal=False):
                if reference is None:
                    st.text('Save a reference result first')
                else:
                    st.altair_chart(diff_chart(reference.ordered_names, current.ordered_names))
                    cols = st.columns(2)
                    for (r,c) in zip([reference, current], cols):
                        with c:
                            if r:
                                st.caption(r.description)
                            else:
                                st.space()
                            
            with st.container(horizontal=False, width="content"):
                st.button(":material/download:", key="dummy", disabled=True, 
                        help="See the chart's own menu to download an image")
                
                st.space()

                def save_ref():
                    st.session_state.undo_reference_result = st.session_state.get('reference_result')
                    st.session_state.reference_result = current
                st.button(':material/arrow_left_alt:', on_click=save_ref, 
                    disabled = (current == reference), help="Save current as reference")
                    
                @st.dialog('Upload Referemce')
                def upload_ref():
                    uploaded_file = st.file_uploader("Choose a result file", 
                        type=["txt", "csv", "result"], 
                        max_upload_size=1, 
                        help="An ordered list of names")
                    text = st.text_area('Or paste an election result here')
                    if uploaded_file:
                        source = 'Uploaded'
                        text = uploaded_file.getvalue().decode("utf-8")
                    else:
                        source = 'Pasted'
                    if text:
                        names = parse_result(text)
                        st.session_state.undo_reference_result = st.session_state.get('reference_result')
                        st.session_state.reference_result = Result(description=source, ordered_names=names)
                        st.rerun()
                st.button(':material/add:', on_click=upload_ref, help="Load a reference result")

                #st.button(':material/content_paste:', on_click=upload_ref, help="Paste a reference result")

                def undo_ref():
                    st.session_state.reference_result = st.session_state.get("undo_reference_result")
                    st.session_state.undo_reference_result = None
                st.button(':material/undo:', on_click=undo_ref, 
                    disabled = (st.session_state.get('undo_reference_result') is None),
                    help="Restore previous reference")
    




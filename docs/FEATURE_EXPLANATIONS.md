# Malware Dataset Feature Explanations

This document explains each feature in the dataset and why it's relevant for malware detection.

---

## Features Used (21 Total After Preprocessing)

### Process State & Priority Features

#### 1. **state**
- **Description**: Current state of the process
- **Type**: Integer code representing process state
- **Malware Relevance**: Malware may exhibit unusual state transitions or stay in specific states (e.g., uninterruptible sleep during rootkit installation)
- **Typical Values**:
  - 0 = Running/Runnable
  - 1 = Interruptible sleep
  - 2 = Uninterruptible sleep
  - 4 = Stopped
  - 8 = Zombie

#### 2. **prio**
- **Description**: Dynamic priority of the process
- **Type**: Integer (lower = higher priority)
- **Malware Relevance**: Malware might try to run at high priority to avoid detection or low priority to hide
- **Normal Range**: 0-139 (0-99 for real-time, 100-139 for normal processes)

#### 3. **static_prio**
- **Description**: Static priority value set when process was created
- **Type**: Integer
- **Malware Relevance**: Malware might manipulate priority to evade detection or ensure resource access
- **Normal Value**: Usually 120 for normal processes

---

### Memory Management Features

#### 4. **vm_truncate_count**
- **Description**: Number of times virtual memory has been truncated
- **Type**: Integer counter
- **Malware Relevance**: Unusual truncation patterns may indicate memory manipulation or code injection
- **Malware Behavior**: Memory-resident malware may frequently modify memory regions

#### 5. **free_area_cache**
- **Description**: Cache pointer for finding free memory areas
- **Type**: Memory address (large integer)
- **Malware Relevance**: Malware manipulating memory layout may show unusual patterns
- **Detection**: Abnormal values suggest memory manipulation techniques

#### 6. **mm_users**
- **Description**: Number of users/threads sharing the memory descriptor
- **Type**: Integer count
- **Malware Relevance**: Malware with multiple threads or process injection shows higher values
- **Normal Range**: 1-10 for typical processes, higher for complex malware

#### 7. **map_count**
- **Description**: Number of memory mappings (VMAs - Virtual Memory Areas)
- **Type**: Integer count
- **Malware Relevance**: High values indicate complex memory usage - common in packed/encrypted malware
- **Detection**: Significantly higher than normal processes suggests malicious activity

---

### Virtual Memory Statistics

#### 8. **total_vm**
- **Description**: Total pages of virtual memory used
- **Type**: Integer (in pages, typically 4KB each)
- **Malware Relevance**: Malware may allocate excessive memory for code unpacking, buffer storage, or C&C data
- **Detection**: Unusually large values suggest memory-intensive malware

#### 9. **shared_vm**
- **Description**: Shared memory pages (shared with other processes)
- **Type**: Integer (pages)
- **Malware Relevance**: Low values suggest isolated malware; high values suggest IPC or shared libraries
- **Detection**: Unusual sharing patterns can indicate process injection

#### 10. **exec_vm**
- **Description**: Executable memory pages (contains code)
- **Type**: Integer (pages)
- **Malware Relevance**: HIGH IMPORTANCE - Malware with packed/encrypted code shows unusual executable memory
- **Detection**: Dynamic code loading/unpacking creates abnormal exec_vm values
- **Why It Matters**: Legitimate programs have stable exec_vm; malware often modifies code at runtime

#### 11. **reserved_vm**
- **Description**: Reserved but not committed memory pages
- **Type**: Integer (pages)
- **Malware Relevance**: Malware may reserve large memory regions for future exploitation
- **Detection**: Excessive reservation without use is suspicious

---

### Process Data Segments

#### 12. **end_data**
- **Description**: End address of the data segment
- **Type**: Memory address (large integer)
- **Malware Relevance**: Malware modifying its data segment shows unusual patterns
- **Detection**: Abnormal data segment sizes suggest runtime modifications
- **Note**: Highly correlated with exec_vm (0.97) - consider removing

---

### Scheduling & Context Switching

#### 13. **last_interval**
- **Description**: Time interval since last scheduling event
- **Type**: Integer (CPU clock ticks)
- **Malware Relevance**: Malware trying to evade detection may sleep/wake in unusual patterns
- **Detection**: Irregular scheduling intervals suggest evasion tactics

#### 14. **nvcsw** (Number of Voluntary Context Switches)
- **Description**: How many times the process voluntarily gave up the CPU
- **Type**: Integer counter
- **Malware Relevance**: HIGH IMPORTANCE - I/O-bound malware (network C&C, file stealing) has high voluntary switches
- **Detection**: Excessive voluntary switches suggest network/file activity
- **Normal**: CPU-bound processes = low; I/O-bound = high

#### 15. **nivcsw** (Number of Involuntary Context Switches)
- **Description**: How many times the process was forcibly preempted by the scheduler
- **Type**: Integer counter
- **Malware Relevance**: CPU-intensive malware (crypto-mining, password cracking) shows high involuntary switches
- **Detection**: High values suggest CPU-hogging behavior

---

### Page Fault Statistics

#### 16. **min_flt** (Minor Page Faults)
- **Description**: Number of page faults resolved without disk I/O
- **Type**: Integer counter
- **Malware Relevance**: Memory-intensive operations cause page faults
- **Detection**: Unpacking/decryption creates burst patterns in minor faults
- **Normal**: Gradual increase; Malware: Sudden spikes

#### 17. **maj_flt** (Major Page Faults)
- **Description**: Number of page faults requiring disk access
- **Type**: Integer counter
- **Malware Relevance**: Loading external modules/libraries causes major faults
- **Detection**: Unusual patterns suggest dynamic code loading
- **Note**: Highly correlated with end_data (0.95) - consider removing

---

### CPU Time Accounting

#### 18. **fs_excl_counter**
- **Description**: Filesystem exclusion counter (for filesystem operations)
- **Type**: Integer counter
- **Malware Relevance**: File-manipulating malware (ransomware, data exfiltration) shows activity
- **Detection**: High values indicate intensive file operations

#### 19. **utime** (User Time)
- **Description**: CPU time spent in user mode (running process code)
- **Type**: Integer (clock ticks)
- **Malware Relevance**: CPU-intensive malware shows high user time
- **Detection**: Disproportionate user time suggests computational tasks (crypto-mining, brute-forcing)
- **Note**: Correlated with nvcsw (0.85) - consider removing

#### 20. **stime** (System Time)
- **Description**: CPU time spent in kernel mode (system calls)
- **Type**: Integer (clock ticks)
- **Malware Relevance**: HIGH IMPORTANCE - System call-heavy malware (rootkits, keyloggers) shows high system time
- **Detection**: Excessive system time suggests kernel interaction or system manipulation
- **Examples**: Rootkits, device drivers, system hooks

#### 21. **gtime** (Guest Time)
- **Description**: CPU time spent running in virtualized guest mode
- **Type**: Integer (clock ticks)
- **Malware Relevance**: Malware detecting/evading virtualization or running in containers
- **Detection**: Non-zero values in non-virtualized environments are suspicious
- **VM Detection**: Some malware checks for virtualization to avoid analysis

---

## Removed Features (11 Constant Features)

These features had zero variance (same value for all samples) and were removed:

1. **usage_counter** - Always 0
2. **normal_prio** - Always 0
3. **policy** - Always 0 (scheduling policy)
4. **vm_pgoff** - Always 0 (virtual memory page offset)
5. **task_size** - Always 0
6. **cached_hole_size** - Always 0
7. **hiwater_rss** - Always 0 (peak resident set size)
8. **nr_ptes** - Always 0 (page table entries)
9. **lock** - Always 0
10. **cgtime** - Always 0 (guest time for child processes)
11. **signal_nvcsw** - Always 0

---

## Feature Importance for Malware Detection

### Top Discriminative Features (Expected)

Based on malware behavior patterns:

#### 🔴 **Critical Features:**
1. **exec_vm** - Malware unpacking/code modification
2. **nvcsw** - Network/file I/O patterns
3. **stime** - System call activity (rootkits, hooks)
4. **map_count** - Complex memory layout (packers)
5. **total_vm** - Excessive memory usage

#### 🟡 **Important Features:**
6. **nivcsw** - CPU-intensive operations
7. **min_flt** - Memory access patterns
8. **fs_excl_counter** - File operations
9. **utime** - Computational intensity
10. **shared_vm** - Process isolation/injection

#### 🟢 **Supporting Features:**
11. **prio** - Priority manipulation
12. **state** - Process state anomalies
13. **mm_users** - Threading patterns
14. **reserved_vm** - Memory reservation tactics
15. **last_interval** - Evasion timing

---

## Malware Behavioral Patterns

### Pattern 1: **Packed/Encrypted Malware**
- **High**: exec_vm, map_count, min_flt
- **Behavior**: Unpacks code at runtime
- **Example**: Trojan droppers, packed executables

### Pattern 2: **Ransomware/File Operations**
- **High**: fs_excl_counter, nvcsw, maj_flt
- **Behavior**: Encrypts files, intensive disk I/O
- **Example**: WannaCry, file-encrypting malware

### Pattern 3: **Network Malware (C&C, Bots)**
- **High**: nvcsw, stime
- **Behavior**: Network communication, waiting on I/O
- **Example**: Botnets, backdoors, RATs

### Pattern 4: **Crypto-Miners**
- **High**: utime, nivcsw, total_vm
- **Behavior**: CPU-intensive computation
- **Example**: XMRig, cryptocurrency miners

### Pattern 5: **Rootkits/System Hooks**
- **High**: stime, map_count, exec_vm
- **Behavior**: Kernel manipulation, system calls
- **Example**: Kernel rootkits, system-level malware

### Pattern 6: **Memory-Resident Malware**
- **High**: total_vm, reserved_vm, mm_users
- **Behavior**: Lives in memory, avoids disk
- **Example**: Fileless malware, memory exploits

---

## Feature Correlations to Watch

### Highly Correlated (Remove for Better Models):
- **exec_vm ↔ end_data** (0.97) → Keep exec_vm, remove end_data
- **maj_flt ↔ end_data** (0.95) → Remove end_data
- **maj_flt ↔ exec_vm** (0.95) → Remove maj_flt
- **utime ↔ nvcsw** (0.85) → Keep nvcsw, remove utime

### Recommended Feature Set (18 features):
After removing highly correlated features:
- state, prio, static_prio, vm_truncate_count, free_area_cache
- mm_users, map_count, total_vm, shared_vm, exec_vm, reserved_vm
- last_interval, nvcsw, nivcsw, min_flt, fs_excl_counter, stime, gtime

---

## How Models Will Use These Features

### Random Forest
- Will identify feature importance automatically
- Expect exec_vm, nvcsw, stime, map_count to rank high
- Can handle non-linear relationships between features

### XGBoost
- Will build decision trees based on feature splits
- Good at capturing complex behavioral patterns
- Feature importance will reveal which metrics matter most

### Neural Network
- Will learn combinations of features
- May discover hidden patterns (e.g., exec_vm + nvcsw + stime = packed network malware)
- Benefits from feature scaling (already done)

---

## Summary

This dataset captures **runtime process behavior** that distinguishes malware from benign software. The features represent:
- **Memory patterns** (how malware uses/manipulates memory)
- **CPU usage** (computational intensity and system interaction)
- **I/O behavior** (file and network activity)
- **Scheduling** (how malware interacts with the OS scheduler)

By analyzing these 21 features, machine learning models can detect malware based on **behavioral signatures** rather than static code analysis, making it effective against new/unknown malware variants.

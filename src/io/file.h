/*
 *  Copyright (C) 2013  Justin Turney
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.

 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-151 USA.
 */

#if !defined(AMBIT_SRC_IO_FILE)
#define AMBIT_SRC_IO_FILE

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

#if defined(DEBUG)
#include <util/print.h>
#endif

namespace ambit { namespace io {

struct file;

enum error {
    kIOErrorOpen = 0,
    kIOErrorReopen,
    kIOErrorClose,
    kIOErrorReclose,
    kIOErrorOStat,
    kIOErrorLSeek,
    kIOErrorRead,
    kIOErrorWrite,
    kIOErrorNoTOCEntry,
    kIOErrorTOCEntrySize,
    kIOErrorKeyLength,
    kIOErrorBlockSize,
    kIOErrorBlockStart,
    kIOErrorBlockEnd
};

enum constants {
    kIOMaxKeyLength = 80,
    kIOPageLength = 64*1024
};

enum OpenMode {
    kOpenModeCreateNew,
    kOpenModeOpenExisting
};

struct address
{
    uint64_t page;
    uint64_t offset;
};

inline bool operator==(const address& a1, const address& a2)
{
    return (a1.page == a2.page &&
            a1.offset == a2.offset);
}

inline bool operator!=(const address& a1, const address& a2)
{
    return !(a1 == a2);
}

inline bool operator>(const address& a1, const address& a2)
{
    if (a1.page > a2.page)
        return true;
    else if (a1.page == a2.page && a1.offset > a2.offset)
        return true;
    return false;
}

inline bool operator<(const address& a1, const address& a2)
{
    if (a1.page < a2.page)
        return true;
    else if (a1.page == a2.page && a1.offset < a2.offset)
        return true;
    return false;
}

namespace util {

address get_address(const address& start, uint64_t shift);

/**
 * @brief Given a start and end address compute the number of bytes
 * between them. Note that end denotes the beginning of the next entry
 * and not the end of the current entry.
 * @param start
 * @param end
 * @return
 */
uint64_t get_length(const address& start, const address& end);

}

namespace toc {

struct entry
{
    char key[kIOMaxKeyLength];
    address start_address;
    address end_address;
};

struct manager
{
    manager(file& owner);

    void initialize();
    void finalize();

    /// Returns the number of entries from the disk
    unsigned int size() const;

    /// Returns the number of bytes the data identified by key is.
    uint64_t size(const std::string& key);

    /// Reads the TOC entries from the file.
    void read();
    /// Writes the TOC entries to the file.
    void write() const;

    /// Print the TOC to the screen.
    void print() const;

    /**
     * @brief Does a specific key exist?
     * @param key The key to search for.
     * @return true, if found; else false.
     */
    bool exists(const std::string& key) const;
    /**
     * @brief For the given key return the entry structure.
     * @param key The key to search for
     * @return The entry struct. If not found will create new entry.
     */
    struct entry& entry(const std::string& key);

    manager(const manager&& other)
        : owner_(other.owner_), contents_(std::move(other.contents_))
    { }

private:
    uint64_t read_size() const;
    void write_size() const;

    file& owner_;
    std::vector<struct entry> contents_;
};

} // namespace toc

struct file
{
    file(const std::string& full_pathname, enum OpenMode om);
//    file(file && other);

    virtual ~file();

    /// Open a file.
    bool open(const std::string& full_pathname, enum OpenMode om);

    /// Close the file.
    void close();

    /// Access the internal handle
    int handle() const { return handle_; }

    /// Access the internal TOC manager
    toc::manager& toc() { return toc_; }
    const toc::manager& toc() const { return toc_; }

    /** Seek to a specific address
     */
    int seek(const address& add) {
        return seek(add.page, add.offset);
    }

    /** Seeks to a specific page and offset in the file.
     * \param page page to go to.
     * \param page offset in the specified page.
     */
    int seek(uint64_t page, uint64_t offset);

    /** Performs a raw read at the address specified by add.
     * \param buffer Memory location to read into.
     * \param add Address to read from.
     * \param count Number of T's to read in.
     */
    template <typename T>
    void read(T* buffer, const address& add, uint64_t count) {
        read_raw(buffer, add, count * sizeof(T));
    }

    /** Performs a raw write at the address specified by add.
     * \param buffer Memory location to read into.
     * \param add Address to read from.
     * \param count Number of T's to read in.
     */
    template <typename T>
    void write(const T* buffer, const address& add, uint64_t count) {
        write_raw(buffer, add, count * sizeof(T));
    }

    /** Performs a write of the data for an entry.
     * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
     * \param buffer Memory location to read into.
     * \param add Address to read from.
     * \param count Number of T's to read in.
     */
    template <typename T>
    void write(const std::string& label, const T* buffer, uint64_t count) {
        // obtain or create the entry in the TOC
        toc::entry& entry = toc_.entry(label);
        // compute location where to start the write
        address write_start = util::get_address(entry.start_address, sizeof(toc::entry));
        // perform a raw write at the location
        write(buffer, write_start, count);
        // update the end_address for the entry.
        entry.end_address = util::get_address(write_start, sizeof(T) * count);
    }

    /** Performs a write of the data for an entry.
     * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
     * \param buffer Memory location to read into.
     * \param add Address to read from.
     * \param count Number of T's to read in.
     */
    template <typename T>
    void read(const std::string& label, T* buffer, uint64_t count) {
        // ensure the entry exists
        if (toc_.exists(label) == false)
            throw std::runtime_error("entry does not exist in the file: " + label);
        // obtain the entry (if we get here it is guarenteed to exist
        toc::entry& entry = toc_.entry(label);
        // compute location where to start the write
        address read_start = util::get_address(entry.start_address, sizeof(toc::entry));
        // perform a raw write at the location
        read(buffer, read_start, count);
        // no update to the entry is performed.
        // check the read count against the end address
        address end = util::get_address(read_start, sizeof(T) * count);
        if (end > entry.end_address)
            throw std::runtime_error("read past the end address of this entry: " + label);
    }

    /** Performs a write of the data for an entry.
     * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
     * \param buffer Memory location to read into.
     * \param add Address to read from.
     * \param count Number of T's to read in.
     */
    template <typename T>
    void write(toc::entry& entry, const T* buffer, uint64_t count) {
        // compute location where to start the write
        address write_start = util::get_address(entry.start_address, sizeof(toc::entry));
        // perform a raw write at the location
        write(buffer, write_start, count);
        // update the end_address for the entry.
        entry.end_address = util::get_address(write_start, sizeof(T) * count);
    }

    /** Performs a write of the data for an entry.
     * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
     * \param buffer Memory location to read into.
     * \param add Address to read from.
     * \param count Number of T's to read in.
     */
    template <typename T>
    void read(const toc::entry& entry, T* buffer, uint64_t count) {
        // compute location where to start the write
        address read_start = util::get_address(entry.start_address, sizeof(toc::entry));
        // perform a raw write at the location
        read(buffer, read_start, count);
        // no update to the entry is performed.
    }

    /** Performs a streaming write of the data for an entry.
     * In practice, the end_address is the start of the write.
     * \param buffer Memory location to read into.
     * \param add Address to read from.
     * \param count Number of T's to read in.
     */
    template <typename T>
    void write_stream(toc::entry& entry, const T* buffer, uint64_t count) {
        // compute location where to start the write
        address write_start = util::get_address(entry.end_address, sizeof(toc::entry));
        // perform a raw write at the location
        write(buffer, write_start, count);
        // update the end_address for the entry.
        entry.end_address = util::get_address(write_start, sizeof(T) * count);
    }

    /** Performs a streaming read of the data for an entry.
     * In practice, the end_address is the start of the write.
     * \param buffer Memory location to read into.
     * \param add Address to read from.
     * \param count Number of T's to read in.
     */
    template <typename T>
    void read_stream(const toc::entry& entry, address& next, const T* buffer, uint64_t count) {
        if (next.page == 0 && next.offset == 0)
            next = util::get_address(entry.start_address, sizeof(toc::entry));

        // compute location where to start the write
        address read_start = next;
        // perform a raw write at the location
        read(buffer, read_start, count);
        // update the end_address for the entry.
        next = util::get_address(read_start, sizeof(T) * count);

        if (next > entry.end_address)
            throw std::runtime_error("read_stream: read beyond the extend of the entry.");
    }

//    file& operator=(file&& other)
//    {
//        handle_ = std::move(other.handle_);
//        name_   = std::move(other.name_);
//        read_stat_ = std::move(other.read_stat_);
//        write_stat_ = std::move(other.write_stat_);
//        toc_ = std::move(other.toc_);
//        return *this;
//    }

    file(file&& other)
        : handle_(other.handle_), name_(other.name_), read_stat_(other.read_stat_), write_stat_(other.write_stat_), toc_(std::move(other.toc_))
    {
        other.handle_ = -1;
    }

protected:

    /** Performs the ultimate reading from the file. Will seek to add and read size number of bytes into buffer.
     */
    void read_raw(void *buffer, const address& add, uint64_t size);

    /** Performs the ultimate writing to the file. Will seek to add and write size number of bytes from buffer.
     */
    void write_raw(const void *buffer, const address& add, uint64_t size);

    /** Used internally to report an error to the user.
     */
    void error(error code);

    /// low-level file handle
    int handle_;

    /// the name of the file
    std::string name_;

    uint64_t read_stat_;
    uint64_t write_stat_;

    toc::manager toc_;

    friend struct toc::manager;
};

}}

#endif

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

#include "file.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cerrno>

#include <util/print.h>

namespace ambit { namespace io {

namespace util {

address get_address(const address& start, uint64_t shift)
{
    address new_address;
    uint64_t bytes_left;

    bytes_left = kIOPageLength - start.offset;

    if (shift >= bytes_left) { // shift to later page
        new_address.page = start.page + (shift - bytes_left) / kIOPageLength + 1;
        new_address.offset = shift - bytes_left - (new_address.page - start.page - 1) * kIOPageLength;
    }
    else { // block starts on current page
        new_address.page = start.page;
        new_address.offset = start.offset + shift;
    }

    return new_address;
}

uint64_t get_length(const address& start, const address& end)
{
    uint64_t full_page_bytes;

    full_page_bytes = (end.page - start.page - 1) * kIOPageLength;

    if (start.page == end.page)
        return end.offset - start.offset;
    else
        return ((kIOPageLength - start.offset) + full_page_bytes + end.offset);
}

}

namespace toc {

manager::manager(file& owner)
    : owner_(owner)
{
}

void manager::initialize()
{

}

void manager::finalize()
{
    write();
}

unsigned int manager::size() const
{
    return contents_.size();
}

bool manager::exists(const std::string& key) const
{
    for (const struct entry& e : contents_) {
        if (key == e.key)
            return true;
    }
    return false;
}

struct entry& manager::entry(const std::string& key)
{
    for (struct entry& e : contents_) {
        if (key == e.key)
            return e;
    }

    // if we get here then we didn't find it.
    // take the last entry and use its end address
    // as our start address.

    // handle special case of no entries.
    struct entry e;
    ::strcpy(e.key, key.c_str());
    if (contents_.size() == 0)
        e.start_address = {0, sizeof(uint64_t)};
    else
        e.start_address = contents_.end()->end_address;
    e.end_address = util::get_address(e.start_address, sizeof(struct entry));

    contents_.push_back(e);
    return contents_.back();
}

uint64_t manager::size(const std::string& key)
{
    if (exists(key)) {
        const struct entry& found = entry(key);
        return util::get_length(found.start_address, found.end_address) - sizeof(struct entry);
    }

    return 0;
}

uint64_t manager::read_size() const
{
    const int handle = owner_.handle();
    int error_code;
    uint64_t len;

    error_code = ::lseek(handle, 0L, SEEK_SET);
    if (error_code == -1)
        owner_.error(kIOErrorLSeek);

    error_code = ::read(handle, &len, sizeof(uint64_t));

    if (error_code != sizeof(uint64_t))
        len = 0;

//    ambit::util::print0("read_length: length %lu\n", len);

    return len;
}

void manager::write_size() const
{
    const int handle = owner_.handle();
    int error_code;

    error_code = ::lseek(handle, 0L, SEEK_SET);
    if (error_code == -1)
        owner_.error(kIOErrorLSeek);

    uint64_t len = contents_.size();
    error_code = ::write(handle, &len, sizeof(uint64_t));
    if (error_code != sizeof(uint64_t))
        owner_.error(kIOErrorWrite);
}

void manager::read()
{
    int entry_size = sizeof(struct entry);
    int handle = owner_.handle();
    uint64_t len = read_size();

    // clear out existing vector
    contents_.clear();
    if (len) {
        address zero = {0, 0};
        address add;

        // start one uint64_t from the start of the file
        add = util::get_address(zero, sizeof(uint64_t));
        for (uint64_t i=0; i<len; ++i) {
            struct entry new_entry;
            owner_.read(&new_entry, add, 1);

//            ambit::util::print0("%-32s %10lu %10lu %10lu %10lu\n",
//                                new_entry.key,
//                                new_entry.start_address.page,
//                                new_entry.start_address.offset,
//                                new_entry.end_address.page,
//                                new_entry.end_address.offset);

            contents_.push_back(new_entry);
            add = new_entry.end_address;
        }
    }
}

void manager::write() const
{
    write_size();

    for (const struct entry& e : contents_) {
        owner_.write(&e, e.start_address, 1);
    }
}

void manager::print() const
{
    ambit::util::print0("----------------------------------------------------------------------------\n");
    ambit::util::print0("Key                                   Spage    Soffset      Epage    Eoffset\n");
    ambit::util::print0("----------------------------------------------------------------------------\n");

    for (const struct entry& e : contents_) {
        ambit::util::print0("%-32s %10lu %10lu %10lu %10lu\n",
                            e.key,
                            e.start_address.page,
                            e.start_address.offset,
                            e.end_address.page,
                            e.end_address.offset);
    }
}

} // namespace toc

file::file(const std::string& full_pathname, enum OpenMode om, enum DeleteMode dm)
    : handle_(-1), name_(full_pathname), read_stat_(0), write_stat_(0), toc_(*this), delete_mode_(dm)
{
    if (open(full_pathname, om) == false)
        throw std::runtime_error("file: Unable to open file " + name_);
}

file::~file()
{
    close();
}

bool file::open(const std::string& full_pathname, enum OpenMode om)
{
    if (handle_ != -1)
        return false;

    name_ = full_pathname;
    if (om == kOpenModeOpenExisting) {
        handle_ = ::open(full_pathname.c_str(), O_CREAT|O_RDWR, 0644);
        if (handle_ == -1)
            throw std::runtime_error("unable to open file: " + full_pathname);
        toc_.read();
    }
    else {
        handle_ = ::open(full_pathname.c_str(), O_CREAT|O_RDWR|O_TRUNC, 0644);
        if (handle_ == -1)
            throw std::runtime_error("unable to open file: " + full_pathname);
        toc_.initialize();
    }

    if (handle_ == -1) // error occurred
        return false;
    return true;
}

void file::close()
{
    if (handle_ != -1) {
        toc_.finalize();
        ::close(handle_);

        if (delete_mode_ == kDeleteModeDeleteOnClose)
            ::unlink(name_.c_str());
    }
    handle_ = -1;
}

void file::error(enum error code)
{
    static const char *error_message[] = {
        "file not open or open call failed",
        "file is already open",
        "file close failed",
        "file is already closed",
        "invalid status flag for file open",
        "lseek failed",
        "error reading from file",
        "error writing to file",
        "no such TOC entry",
        "TOC entry size mismatch",
        "TOC key too long",
        "requested block size is invalid",
        "incorrect block start address",
        "incorrect block end address",
    };

    ambit::util::printn("io error: %d, %s; errno %d\n",
                        code,
                        error_message[code],
                        errno);
    ambit::util::print0("io error: %d, %s; errno %d\n",
                        code,
                        error_message[code],
                        errno);

    ::exit(EXIT_FAILURE);
}

int file::seek(uint64_t page, uint64_t offset)
{
    // this is strictly to avoid overflow errors on lseek calls
    const uint64_t bignum = 10000;
    int error_code;
    uint64_t total_offset;

    // move to the beginning
    error_code = ::lseek(handle_, 0, SEEK_SET);

    if (error_code == -1)
        return error_code;

    // lseek through large chunks of the file to avoid offset overflows
    for (; page > bignum; page -= bignum) {
        total_offset = bignum * kIOPageLength;
        error_code = ::lseek(handle_, total_offset, SEEK_CUR);
        if (error_code == -1)
            return error_code;
    }

    // now compute the file offset including the page-relative term
    total_offset = page;
    total_offset *= kIOPageLength;
    total_offset += offset; // add the page-relative term

//    ambit::util::print0("seeking to %lu (page %lu, offset %lu, page length %lu)\n",
//                        total_offset, page, offset, kIOPageLength);

    error_code = ::lseek(handle_, total_offset, SEEK_CUR);
    if (error_code == -1)
        return error_code;

    return 0;
}

void file::read_raw(void *buffer, const address& add, uint64_t size)
{
    uint64_t error_code;

    // seek to the needed address
    seek(add);

    error_code = ::read(handle_, buffer, size);
    if (error_code != size)
        error(kIOErrorRead);

    read_stat_ += size;
}

void file::write_raw(const void *buffer, const address& add, uint64_t size)
{
    uint64_t error_code;

    // seek to the needed address
    seek(add);

    error_code = ::write(handle_, buffer, size);
    if (error_code != size)
        error(kIOErrorRead);

    write_stat_ += size;
}

}}

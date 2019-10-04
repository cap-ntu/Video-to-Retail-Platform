import React from "react";
import * as PropTypes from "prop-types";
import Table from "@material-ui/core/Table";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import TableCell from "@material-ui/core/TableCell";
import TableBody from "@material-ui/core/TableBody";
import withStyles from "@material-ui/core/styles/withStyles";
import Avatar from "@material-ui/core/Avatar";

const styles = {};

const DetectionTable = ({classes, rows}) => (
    <Table className={classes.table}>
        <TableHead>
            <TableRow>
                <TableCell>Image</TableCell>
                <TableCell align="center">Name</TableCell>
                <TableCell align="right">Confidence %</TableCell>
            </TableRow>
        </TableHead>
        <TableBody>
            {rows.map((row, index) => (
                <TableRow key={index}>
                    <TableCell component="th" scope="row">
                        <Avatar src={row.src || `https://api.adorable.io/avatars/285/${row.name}`}/>
                    </TableCell>
                    <TableCell align="center">{row.name}</TableCell>
                    {
                        row.score ?
                            <TableCell align="right">{Math.round(row.score * 10000) / 100}</TableCell> :
                            <TableCell align="center">N.A.</TableCell>
                    }
                </TableRow>
            ))}
        </TableBody>
    </Table>
);

DetectionTable.defaultProps = {
    rows: [],
};

DetectionTable.propTypes = {
    classes: PropTypes.object.isRequired,
    rows: PropTypes.array.isRequired,
};

export default withStyles(styles)(DetectionTable);
